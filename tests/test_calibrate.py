import json

from loopcheck.calibrate import run_calibration, save_report
from loopcheck.config import Config


def test_calibration_no_judge_separates_good_from_bad(tmp_path):
    cfg = Config(max_mutants=8)
    report = run_calibration(cfg, llm=None)
    assert report.n == 15
    good_rows = [r for r in report.rows if r["label"] == "good"]
    bad_rows = [r for r in report.rows if r["label"] == "bad"]
    mean_good = sum(r["confidence"] for r in good_rows) / len(good_rows)
    mean_bad = sum(r["confidence"] for r in bad_rows) / len(bad_rows)
    assert mean_good > mean_bad  # the verifier must at least rank good above bad
    assert 0.0 <= report.precision <= 1.0 and 0.0 <= report.recall <= 1.0
    assert sum(b["count"] for b in report.bins) == report.n

    out = tmp_path / "report.json"
    save_report(report, out)
    data = json.loads(out.read_text())
    assert data["n"] == 15 and len(data["rows"]) == 15
