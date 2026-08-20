"""Regenerate calibration/demo_baseline.json for `loopcheck demo`.

`loopcheck demo` (default, no --with-judge) prints real verifier output without
making any subprocess or network calls at demo time, so it is fast and
reproducible in front of an audience. The numbers it prints are read from this
file instead of being recomputed live. This script is how you refresh them
after changing a target, a check, or the calibration set.

Run: uv run python scripts/gen_demo_baseline.py
"""

import json
from pathlib import Path

from loopcheck.calibrate import run_calibration
from loopcheck.config import Config
from loopcheck.target import load_target
from loopcheck.verifier import verify

DEMO_TARGET = "slugify"
DEMO_FILE = Path("calibration/slugify/good__thorough.py")
OUT_PATH = Path("calibration/demo_baseline.json")


def main() -> None:
    config = Config()
    target = load_target(Path("targets") / DEMO_TARGET)
    test_code = DEMO_FILE.read_text()
    verdict = verify(test_code, target, llm=None, config=config, history=[])

    report = run_calibration(config, llm=None)

    data = {
        "generated_by": "scripts/gen_demo_baseline.py",
        "note": "Real, pre-computed verifier output. `loopcheck demo` reads this "
        "file so the default demo has no subprocess or network calls. "
        "Regenerate with `uv run python scripts/gen_demo_baseline.py`.",
        "verify": {
            "target": DEMO_TARGET,
            "file": str(DEMO_FILE),
            "confidence": round(verdict.confidence, 4),
            "decision": verdict.decision,
            "checks": [
                {"name": c.name, "score": round(c.score, 4)} for c in verdict.checks
            ],
        },
        "calibrate": {
            "n": report.n,
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "f1": round(report.f1, 4),
        },
    }
    OUT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
