import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from loopcheck.audit import append_record
from loopcheck.config import Config
from loopcheck.db import connect
from loopcheck.llm import LLM
from loopcheck.target import Target
from loopcheck.tracing import init_tracing
from loopcheck.verifier import verify


class LoopState(TypedDict):
    run_id: str
    target_name: str
    test_code: str
    feedback: str
    iteration: int
    confidence_history: list[float]
    decision: str
    total_cost_usd: float


_GENERATE_SYSTEM = """You write high-quality pytest test files. You will receive a module's \
spec and source. Write a single self-contained pytest file that verifies the INTENDED behavior \
in the spec, including edge cases and error conditions. Import the module by its module name. \
Use exact-value assertions. Do not mock the module under test. \
Respond with ONLY a fenced python code block."""


def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() + "\n" if m else text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_generate_prompt(target: Target, state: LoopState) -> str:
    prompt = (
        f"## Module name\n{target.module_name}\n\n## Spec\n{target.spec}\n\n"
        f"## Source\n```python\n{target.source}\n```"
    )
    if state["test_code"]:
        prompt += (
            f"\n\n## Your previous attempt\n```python\n{state['test_code']}\n```"
            f"\n\n## Verifier feedback (fix these issues)\n{state['feedback']}"
        )
    return prompt


def run_loop(
    target: Target,
    config: Config,
    llm: LLM,
    run_id: str,
    checkpoint_path: str = "checkpoints.db",
    resume: bool = False,
) -> LoopState:
    conn = connect(config.db_path)
    tracer = init_tracing(conn, config.otlp_endpoint)
    audit_key = config.audit_key()

    def generate(state: LoopState) -> dict:
        with tracer.start_as_current_span(
            "generate",
            attributes={"loopcheck.run_id": run_id, "iteration": state["iteration"] + 1},
        ) as span:
            resp = llm.complete(
                _GENERATE_SYSTEM, _build_generate_prompt(target, state), config.agent_model
            )
            span.set_attribute("cost_usd", resp.cost_usd)
            span.set_attribute("output_tokens", resp.output_tokens)
            return {
                "test_code": extract_code_block(resp.text),
                "iteration": state["iteration"] + 1,
                "total_cost_usd": state["total_cost_usd"] + resp.cost_usd,
            }

    def verify_node(state: LoopState) -> dict:
        with tracer.start_as_current_span(
            "verify",
            attributes={"loopcheck.run_id": run_id, "iteration": state["iteration"]},
        ) as span:
            history = state["confidence_history"]
            verdict = verify(state["test_code"], target, llm, config, history)
            for c in verdict.checks:
                with tracer.start_as_current_span(
                    f"check:{c.name}", attributes={"score": c.score, "detail": c.detail[:500]}
                ):
                    pass
            span.set_attribute("confidence", verdict.confidence)
            span.set_attribute("decision", verdict.decision)
            span.set_attribute("cost_usd", verdict.cost_usd)
            rationale = verdict.judge.rationale if verdict.judge else ""
            append_record(
                conn, audit_key, run_id, state["iteration"],
                {
                    "run_id": run_id,
                    "iteration": state["iteration"],
                    "ts": _now(),
                    "confidence": round(verdict.confidence, 4),
                    "decision": verdict.decision,
                    "checks": {c.name: round(c.score, 4) for c in verdict.checks},
                    "rationale_sha256": sha256(rationale.encode()).hexdigest(),
                },
            )
            return {
                "confidence_history": history + [verdict.confidence],
                "decision": verdict.decision,
                "feedback": verdict.feedback,
                "total_cost_usd": state["total_cost_usd"] + verdict.cost_usd,
            }

    def route(state: LoopState) -> str:
        if state["decision"] == "retry" and state["iteration"] < config.max_iterations:
            return "generate"
        return END

    graph = StateGraph(LoopState)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route)

    initial: LoopState = {
        "run_id": run_id,
        "target_name": target.name,
        "test_code": "",
        "feedback": "",
        "iteration": 0,
        "confidence_history": [],
        "decision": "",
        "total_cost_usd": 0.0,
    }
    if not resume:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, 'running', ?, NULL, 0, NULL, 0)",
            (run_id, target.name, _now()),
        )
        conn.commit()

    with SqliteSaver.from_conn_string(checkpoint_path) as saver:
        compiled = graph.compile(checkpointer=saver)
        thread = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        final = compiled.invoke(None if resume else initial, thread)

    status = final["decision"]
    if status == "retry":  # hit the iteration cap while still retrying
        status = "unconverged"
    conn.execute(
        "UPDATE runs SET status=?, finished_ts=?, iterations=?, final_confidence=?, "
        "cost_usd=? WHERE run_id=?",
        (status, _now(), final["iteration"],
         final["confidence_history"][-1] if final["confidence_history"] else None,
         final["total_cost_usd"], run_id),
    )
    conn.commit()
    conn.close()
    return final  # type: ignore[return-value]
