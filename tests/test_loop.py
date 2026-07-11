import json

import pytest

from loopcheck.audit import show_run, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect
from loopcheck.llm import FakeLLM
from loopcheck.loop import extract_code_block, run_loop
from loopcheck.target import Target

TARGET = Target(
    name="mymod", module_name="mymod",
    source="def double(x):\n    if x < 0:\n        return -2 * x\n    return 2 * x\n",
    spec="double(x): 2*x for x>=0, -2*x for x<0. Always non-negative doubling magnitude.",
)

GOOD_TESTS = """```python
from mymod import double

def test_pos():
    assert double(3) == 6

def test_neg():
    assert double(-3) == 6

def test_zero():
    assert double(0) == 0
```"""

BAD_TESTS = """```python
from mymod import double

def test_pos():
    assert double(3) == 6
```"""

PERFECT_JUDGE = json.dumps({
    "criteria": {"intent": 1.0, "edge_cases": 1.0, "assertion_quality": 1.0,
                 "independence": 1.0},
    "rationale": "thorough",
})
HARSH_JUDGE = json.dumps({
    "criteria": {"intent": 0.2, "edge_cases": 0.0, "assertion_quality": 0.1,
                 "independence": 1.0},
    "rationale": "only the positive path is tested",
})
# Expected confidence with BAD_TESTS ≈ 0.61 (retry band), GOOD_TESTS ≈ 0.91 (accept):
# BAD kills ~2/5 mutants (positive-path only), GOOD kills ~4/5 plus perfect judge.


def test_extract_code_block():
    assert extract_code_block(GOOD_TESTS).startswith("from mymod import double")
    assert extract_code_block("plain code") == "plain code"


def test_loop_converges_second_iteration(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db", max_iterations=5)
    llm = FakeLLM([BAD_TESTS, HARSH_JUDGE, GOOD_TESTS, PERFECT_JUDGE])
    state = run_loop(TARGET, cfg, llm, run_id="run-x",
                     checkpoint_path=str(tmp_path / "ckpt.db"))
    assert state["decision"] == "accept"
    assert state["iteration"] == 2
    assert len(state["confidence_history"]) == 2
    assert state["confidence_history"][1] > state["confidence_history"][0]

    conn = connect(cfg.db_path)
    run = conn.execute("SELECT * FROM runs WHERE run_id='run-x'").fetchone()
    assert run["status"] == "accept" and run["iterations"] == 2

    records = show_run(conn, "run-x")
    assert [r["decision"] for r in records] == ["retry", "accept"]
    assert verify_chain(conn, cfg.audit_key()) == (True, None)

    span_names = {r["name"] for r in conn.execute("SELECT name FROM spans")}
    assert {"generate", "verify", "check:mutation"} <= span_names


def test_loop_feedback_reaches_next_generation(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db")
    llm = FakeLLM([BAD_TESTS, HARSH_JUDGE, GOOD_TESTS, PERFECT_JUDGE])
    run_loop(TARGET, cfg, llm, run_id="run-y", checkpoint_path=str(tmp_path / "ckpt.db"))
    second_generate_prompt = llm.calls[2][1]
    assert "only the positive path is tested" in second_generate_prompt  # judge rationale fed back


def test_loop_stops_at_max_iterations(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db", max_iterations=2)
    llm = FakeLLM([BAD_TESTS, HARSH_JUDGE] * 2)
    state = run_loop(TARGET, cfg, llm, run_id="run-z",
                     checkpoint_path=str(tmp_path / "ckpt.db"))
    assert state["iteration"] == 2
    assert state["decision"] in ("retry", "escalate")
    conn = connect(cfg.db_path)
    run = conn.execute("SELECT * FROM runs WHERE run_id='run-z'").fetchone()
    assert run["status"] in ("unconverged", "escalate")


def test_loop_resume_preserves_run_row(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db", max_iterations=5)
    ckpt = str(tmp_path / "ckpt.db")

    class ExplodingLLM:
        def __init__(self):
            self.inner = FakeLLM([BAD_TESTS, HARSH_JUDGE])
            self.calls = self.inner.calls
        def complete(self, system, user, model, max_tokens=4096):
            if not self.inner.responses:
                raise RuntimeError("boom")
            return self.inner.complete(system, user, model, max_tokens)

    with pytest.raises(RuntimeError):
        run_loop(TARGET, cfg, ExplodingLLM(), run_id="run-r", checkpoint_path=ckpt)

    conn = connect(cfg.db_path)
    started = conn.execute("SELECT started_ts FROM runs WHERE run_id='run-r'").fetchone()[0]
    conn.close()

    # LangGraph replays the interrupted node from the checkpoint; the resumed run
    # picks up from where it crashed. The verify node already completed before the
    # crash (HARSH_JUDGE was consumed), so resume continues with a fresh generate
    # call followed by verify. We provide GOOD_TESTS + PERFECT_JUDGE for that.
    state = run_loop(TARGET, cfg, FakeLLM([GOOD_TESTS, PERFECT_JUDGE]), run_id="run-r",
                     checkpoint_path=ckpt, resume=True)
    assert state["decision"] == "accept"
    conn = connect(cfg.db_path)
    row = conn.execute("SELECT started_ts, status FROM runs WHERE run_id='run-r'").fetchone()
    assert row["started_ts"] == started  # resume must not clobber the original start
    assert row["status"] == "accept"
