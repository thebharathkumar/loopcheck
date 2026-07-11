import pytest

from loopcheck.llm import FakeLLM, cost_usd


def test_cost_known_model():
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert cost_usd("claude-haiku-4-5-20251001", 2_000_000, 0) == pytest.approx(2.0)


def test_cost_unknown_model():
    assert cost_usd("gpt-x", 1000, 1000) == 0.0


def test_fake_llm_pops_and_records():
    fake = FakeLLM(["first", "second"])
    r1 = fake.complete("sys", "hello", "any-model")
    assert r1.text == "first" and r1.cost_usd == 0.0
    assert fake.complete("sys", "again", "any-model").text == "second"
    assert fake.calls == [("sys", "hello"), ("sys", "again")]
    with pytest.raises(IndexError):
        fake.complete("sys", "third", "any-model")
