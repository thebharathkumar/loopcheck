from dataclasses import dataclass

from loopcheck.checks import (
    CheckResult,
    check_assertions,
    check_coverage,
    check_mutation,
    check_no_target_mock,
    check_tests_pass,
)
from loopcheck.config import Config
from loopcheck.judge import JudgeResult, judge_tests
from loopcheck.llm import LLM
from loopcheck.target import Target


@dataclass
class Verdict:
    confidence: float
    decision: str
    checks: list[CheckResult]
    judge: JudgeResult | None
    feedback: str
    cost_usd: float


def compute_confidence(
    checks: list[CheckResult], judge: JudgeResult | None, weights: dict[str, float]
) -> float:
    scores = {c.name: c.score for c in checks}
    if judge is not None:
        scores["judge"] = judge.score
    present = {k: w for k, w in weights.items() if k in scores}
    total = sum(present.values())
    if total == 0:
        return 0.0
    return sum(scores[k] * w for k, w in present.items()) / total


def decide(confidence: float, history: list[float], config: Config) -> str:
    if confidence >= config.accept_threshold:
        return "accept"
    if confidence < config.escalate_threshold:
        return "escalate"
    if len(history) >= 2 and confidence <= history[-1] <= history[-2]:
        return "escalate"  # stagnating: two consecutive non-improvements
    return "retry"


def synthesize_feedback(checks: list[CheckResult], judge: JudgeResult | None) -> str:
    parts = [f"[{c.name}] score={c.score:.2f}: {c.detail}" for c in checks if c.score < 1.0]
    if judge is not None and judge.score < 0.9:
        parts.append(f"[judge] score={judge.score:.2f}: {judge.rationale}")
    return "\n".join(parts) if parts else "all checks passed"


def verify(
    test_code: str,
    target: Target,
    llm: LLM | None,
    config: Config,
    history: list[float],
) -> Verdict:
    tests_pass = check_tests_pass(
        test_code, target.source, target.module_name, config.test_timeout_s
    )
    if tests_pass.score == 0.0:
        coverage = CheckResult("coverage", 0.0, "skipped: tests fail on unmutated module")
        mutation = CheckResult("mutation", 0.0, "skipped: tests fail on unmutated module")
    else:
        coverage = check_coverage(
            test_code, target.source, target.module_name, config.test_timeout_s
        )
        mutation = check_mutation(
            test_code, target.source, target.module_name,
            config.max_mutants, config.test_timeout_s,
        )
    checks = [
        tests_pass,
        check_assertions(test_code),
        check_no_target_mock(test_code, target.module_name),
        coverage,
        mutation,
    ]
    judge = None
    if llm is not None:
        judge = judge_tests(llm, config.judge_model, target.spec, target.source, test_code)
    confidence = compute_confidence(checks, judge, config.weights)
    return Verdict(
        confidence=confidence,
        decision=decide(confidence, history, config),
        checks=checks,
        judge=judge,
        feedback=synthesize_feedback(checks, judge),
        cost_usd=judge.cost_usd if judge else 0.0,
    )
