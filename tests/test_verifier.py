import json

import pytest

from loopcheck.checks import CheckResult
from loopcheck.config import Config
from loopcheck.judge import JudgeResult
from loopcheck.llm import FakeLLM
from loopcheck.target import Target
from loopcheck.verifier import compute_confidence, decide, synthesize_feedback, verify

CFG = Config()


def _checks(mutation=1.0, tests_pass=1.0, assertions=1.0, no_target_mock=1.0, coverage=1.0):
    return [
        CheckResult("mutation", mutation, "m"),
        CheckResult("tests_pass", tests_pass, "t"),
        CheckResult("assertions", assertions, "a"),
        CheckResult("no_target_mock", no_target_mock, "n"),
        CheckResult("coverage", coverage, "c"),
    ]


def _judge(score):
    return JudgeResult(score, {"intent": score}, "rationale text", 0.0)


def test_confidence_perfect():
    assert compute_confidence(_checks(), _judge(1.0), CFG.weights) == pytest.approx(1.0)


def test_confidence_weighted():
    conf = compute_confidence(_checks(mutation=0.0), _judge(1.0), CFG.weights)
    assert conf == pytest.approx(0.55)  # lost the 0.45 mutation weight


def test_confidence_without_judge_renormalizes():
    conf = compute_confidence(_checks(), None, CFG.weights)
    assert conf == pytest.approx(1.0)


def test_decide_thresholds():
    assert decide(0.90, [], CFG) == "accept"
    assert decide(0.60, [], CFG) == "retry"
    assert decide(0.40, [], CFG) == "retry"  # first attempt always retries unless accepted


def test_decide_stagnation_escalates():
    assert decide(0.60, [0.70, 0.65], CFG) == "escalate"   # two non-improvements
    assert decide(0.70, [0.50, 0.65], CFG) == "retry"      # still improving


def test_decide_first_attempt_never_escalates_on_low_confidence():
    assert decide(0.18, [], CFG) == "retry"     # failing first draft gets feedback
    assert decide(0.18, [0.20], CFG) == "escalate"  # second low attempt escalates


def test_decide_stagnation_equal_confidences():
    assert decide(0.60, [0.60, 0.60], CFG) == "escalate"


def test_feedback_mentions_failures_only():
    checks = _checks(mutation=0.4)
    checks[0] = CheckResult("mutation", 0.4, "survivors: [off_by_one at line 3]")
    fb = synthesize_feedback(checks, _judge(0.5))
    assert "off_by_one" in fb and "rationale text" in fb
    assert "no_target_mock" not in fb


def test_verify_end_to_end_with_fake_judge():
    target = Target(
        name="mymod", module_name="mymod",
        source="def double(x):\n    if x < 0:\n        return -2 * x\n    return 2 * x\n",
        spec="double(x): returns 2*x for x>=0, -2*x for x<0",
    )
    tests = (
        "from mymod import double\n"
        "def test_pos():\n    assert double(3) == 6\n"
        "def test_neg():\n    assert double(-3) == 6\n"
        "def test_zero():\n    assert double(0) == 0\n"
    )
    judge_json = json.dumps({
        "criteria": {"intent": 1.0, "edge_cases": 1.0, "assertion_quality": 1.0,
                     "independence": 1.0},
        "rationale": "solid",
    })
    v = verify(tests, target, FakeLLM([judge_json]), CFG, history=[])
    assert 0.0 < v.confidence <= 1.0
    assert v.decision in ("accept", "retry", "escalate")
    assert {c.name for c in v.checks} == {
        "mutation", "tests_pass", "assertions", "no_target_mock", "coverage"
    }


def test_verify_without_llm_skips_judge():
    target = Target("mymod", "mymod", "def one():\n    return 1\n", "one() returns 1")
    v = verify("from mymod import one\ndef test_one():\n    assert one() == 1\n",
               target, None, CFG, history=[])
    assert v.judge is None and v.confidence > 0.0


def test_verify_failing_tests_zero_mutation_score():
    target = Target(
        name="mymod", module_name="mymod",
        source="def double(x):\n    if x < 0:\n        return -2 * x\n    return 2 * x\n",
        spec="double(x): 2*x for x>=0, -2*x for x<0",
    )
    wrong = "from mymod import double\ndef test_wrong():\n    assert double(3) == 7\n"
    v = verify(wrong, target, None, CFG, history=[])
    scores = {c.name: c.score for c in v.checks}
    assert scores["tests_pass"] == 0.0
    assert scores["mutation"] == 0.0
    assert scores["coverage"] == 0.0
    assert "skipped" in next(c for c in v.checks if c.name == "mutation").detail
