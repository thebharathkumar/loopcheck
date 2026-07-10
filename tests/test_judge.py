import json

import pytest

from loopcheck.judge import judge_tests
from loopcheck.llm import FakeLLM

GOOD_JSON = json.dumps({
    "criteria": {"intent": 1.0, "edge_cases": 0.5, "assertion_quality": 1.0, "independence": 0.5},
    "rationale": "covers happy path, misses unicode edge cases",
})


def test_judge_parses_scores():
    fake = FakeLLM([GOOD_JSON])
    r = judge_tests(fake, "m", "SPEC", "def f(): pass", "def test_f(): assert True")
    assert r.score == pytest.approx(0.75)
    assert r.criteria["edge_cases"] == 0.5
    assert "unicode" in r.rationale


def test_judge_extracts_json_from_chatter():
    fake = FakeLLM(["Here is my assessment:\n" + GOOD_JSON + "\nHope that helps!"])
    assert judge_tests(fake, "m", "s", "src", "tests").score == pytest.approx(0.75)


def test_judge_prompt_contains_inputs():
    fake = FakeLLM([GOOD_JSON])
    judge_tests(fake, "m", "THE_SPEC", "THE_SOURCE", "THE_TESTS")
    _, user = fake.calls[0]
    assert "THE_SPEC" in user and "THE_SOURCE" in user and "THE_TESTS" in user


def test_judge_rejects_garbage():
    with pytest.raises(ValueError):
        judge_tests(FakeLLM(["no json here"]), "m", "s", "src", "tests")
