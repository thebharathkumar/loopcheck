import io
import json
import time

import pytest

from loopcheck.config import Config
from loopcheck.demo import DemoError, run_demo
from loopcheck.llm import FakeLLM

FAKE_BASELINE = {
    "verify": {
        "target": "slugify",
        "file": "calibration/slugify/good__thorough.py",
        "confidence": 1.0,
        "decision": "accept",
        "checks": [
            {"name": "tests_pass", "score": 1.0},
            {"name": "assertions", "score": 1.0},
            {"name": "no_target_mock", "score": 1.0},
            {"name": "coverage", "score": 1.0},
            {"name": "mutation", "score": 1.0},
        ],
    },
    "calibrate": {"n": 15, "precision": 1.0, "recall": 0.3333, "f1": 0.5},
}


def _write_baseline(tmp_path, monkeypatch, data=FAKE_BASELINE):
    p = tmp_path / "demo_baseline.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr("loopcheck.demo.BASELINE_PATH", p)
    return p


def test_default_mode_prints_all_three_stages_and_succeeds(tmp_path, monkeypatch):
    _write_baseline(tmp_path, monkeypatch)
    out = io.StringIO()
    assert run_demo(with_judge=False, out=out) == 0
    text = out.getvalue()
    assert "[1/3] verify" in text
    assert "ACCEPT" in text
    assert "[2/3] calibrate" in text
    assert "precision=1.00" in text and "recall=0.33" in text
    assert "[3/3] audit" in text
    assert "HMAC-SHA256 chain intact -- 1 record verified" in text
    assert "done in" in text


def test_default_mode_makes_no_llm_call(tmp_path, monkeypatch):
    _write_baseline(tmp_path, monkeypatch)
    fake = FakeLLM([])  # would raise IndexError if .complete() were ever called
    assert run_demo(with_judge=False, llm=fake, out=io.StringIO()) == 0
    assert fake.calls == []


def test_default_mode_is_fast(tmp_path, monkeypatch):
    _write_baseline(tmp_path, monkeypatch)
    t0 = time.monotonic()
    assert run_demo(with_judge=False, out=io.StringIO()) == 0
    assert time.monotonic() - t0 < 2.0


def test_default_mode_repeated_runs_produce_identical_output(tmp_path, monkeypatch):
    _write_baseline(tmp_path, monkeypatch)

    def _lines_without_timing():
        out = io.StringIO()
        run_demo(with_judge=False, out=out)
        return [ln for ln in out.getvalue().splitlines() if not ln.startswith("done in")]

    first = _lines_without_timing()
    for _ in range(2):
        assert _lines_without_timing() == first


def test_missing_baseline_fails_cleanly_no_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr("loopcheck.demo.BASELINE_PATH", tmp_path / "nope.json")
    out = io.StringIO()
    assert run_demo(with_judge=False, out=out) == 1
    text = out.getvalue()
    assert "[1/3]" not in text and "[2/3]" not in text and "[3/3]" not in text


def test_corrupt_baseline_fails_cleanly(tmp_path, monkeypatch):
    p = tmp_path / "demo_baseline.json"
    p.write_text("{not json")
    monkeypatch.setattr("loopcheck.demo.BASELINE_PATH", p)
    assert run_demo(with_judge=False, out=io.StringIO()) == 1


def test_with_judge_calls_llm_and_reports_judge_score(tmp_path, monkeypatch):
    _write_baseline(tmp_path, monkeypatch)
    good_json = json.dumps(
        {
            "criteria": {
                "intent": 1.0, "edge_cases": 1.0,
                "assertion_quality": 1.0, "independence": 1.0,
            },
            "rationale": "solid",
        }
    )
    fake = FakeLLM([good_json])
    out = io.StringIO()
    rc = run_demo(with_judge=True, llm=fake, config=Config(max_mutants=3), out=out)
    assert rc == 0
    text = out.getvalue()
    assert "judge=1.00" in text
    assert "live, calling judge" in text
    assert fake.calls  # the judge really was called


def test_with_judge_llm_failure_is_one_clean_line_no_traceback():
    class BoomLLM:
        def complete(self, *a, **k):
            raise RuntimeError("connection refused")

    out = io.StringIO()
    err = io.StringIO()
    import sys as _sys

    old_stderr = _sys.stderr
    _sys.stderr = err
    try:
        rc = run_demo(with_judge=True, llm=BoomLLM(), config=Config(max_mutants=3), out=out)
    finally:
        _sys.stderr = old_stderr
    assert rc == 1
    assert "Traceback" not in err.getvalue()
    assert "connection refused" in err.getvalue()
    assert err.getvalue().count("\n") == 1  # exactly one clean error line


def test_demo_error_message_mentions_regenerate_command(tmp_path, monkeypatch):
    monkeypatch.setattr("loopcheck.demo.BASELINE_PATH", tmp_path / "nope.json")
    with pytest.raises(DemoError, match="gen_demo_baseline"):
        from loopcheck.demo import _load_baseline

        _load_baseline()
