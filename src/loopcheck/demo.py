"""`loopcheck demo` — the full verify -> calibrate -> audit pipeline, in one
command, meant to be run live in front of an audience.

Default mode reads real, pre-computed verifier output from
calibration/demo_baseline.json (see scripts/gen_demo_baseline.py) instead of
re-running mutation testing and calibration live: those are real subprocess-
and LLM-heavy operations that take seconds to minutes and would make a live
demo slow and non-reproducible. This mode makes no network or subprocess
calls and is deterministic run to run.

--with-judge runs the real verifier live, including the LLM judge call, and
requires ANTHROPIC_API_KEY.
"""

import json
import sys
import time
from pathlib import Path

from loopcheck.audit import append_record, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect
from loopcheck.llm import LLM
from loopcheck.target import load_target
from loopcheck.verifier import verify

BASELINE_PATH = Path("calibration/demo_baseline.json")


class DemoError(Exception):
    """Carries a single clean line for the CLI to print on failure."""


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        raise DemoError(
            f"{BASELINE_PATH} is missing -- regenerate with: "
            "uv run python scripts/gen_demo_baseline.py"
        )
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise DemoError(f"could not read {BASELINE_PATH}: {e}") from e


def _run_verify_stage(with_judge: bool, llm: LLM | None, config: Config, out) -> dict:
    baseline = _load_baseline()
    v = baseline["verify"]

    if not with_judge:
        print(f"[1/3] verify      {v['target']}/{Path(v['file']).name}", file=out)
        checks_line = "  ".join(f"{c['name']}={c['score']:.2f}" for c in v["checks"])
        print(f"          {checks_line}", file=out)
        print(
            f"          confidence {v['confidence']:.3f} -> {v['decision'].upper()}",
            file=out,
        )
        return {
            "target": v["target"],
            "confidence": v["confidence"],
            "decision": v["decision"],
            "checks": {c["name"]: c["score"] for c in v["checks"]},
        }

    target = load_target(Path("targets") / v["target"])
    test_code = Path(v["file"]).read_text()
    print(f"[1/3] verify      {v['target']}/{Path(v['file']).name}  (live, calling judge)", file=out)
    try:
        verdict = verify(test_code, target, llm, config, history=[])
    except Exception as e:  # judge/network failure must surface as one clean line
        raise DemoError(f"judge call failed: {e}") from e
    checks_line = "  ".join(f"{c.name}={c.score:.2f}" for c in verdict.checks)
    if verdict.judge is not None:
        checks_line += f"  judge={verdict.judge.score:.2f}"
    print(f"          {checks_line}", file=out)
    print(
        f"          confidence {verdict.confidence:.3f} -> {verdict.decision.upper()}",
        file=out,
    )
    return {
        "target": v["target"],
        "confidence": verdict.confidence,
        "decision": verdict.decision,
        "checks": {c.name: c.score for c in verdict.checks},
    }


def _run_calibrate_stage(out) -> None:
    baseline = _load_baseline()
    c = baseline["calibrate"]
    print(f"[2/3] calibrate   {c['n']} labeled files (slugify, pricing, ratelimit)", file=out)
    print(
        f"          precision={c['precision']:.2f}  recall={c['recall']:.2f}  f1={c['f1']:.2f}",
        file=out,
    )


def _run_audit_stage(verify_result: dict, out) -> None:
    config = Config()
    key = config.audit_key()
    conn = connect(":memory:")
    try:
        append_record(
            conn,
            key,
            "demo",
            1,
            {
                "run_id": "demo",
                "target": verify_result["target"],
                "confidence": round(verify_result["confidence"], 4),
                "decision": verify_result["decision"],
                "checks": {k: round(v, 4) for k, v in verify_result["checks"].items()},
            },
        )
        ok, bad_seq = verify_chain(conn, key)
        n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    finally:
        conn.close()
    if not ok:
        raise DemoError(f"audit chain BROKEN at seq {bad_seq}")
    record = "record" if n == 1 else "records"
    print(f"[3/3] audit       HMAC-SHA256 chain intact -- {n} {record} verified", file=out)


def run_demo(
    with_judge: bool = False, llm: LLM | None = None, config: Config | None = None, out=None
) -> int:
    out = out if out is not None else sys.stdout
    config = config if config is not None else Config()
    mode = "with LLM judge (live API call)" if with_judge else "deterministic, no API calls"
    print(f"loopcheck demo -- {mode}", file=out)
    t0 = time.monotonic()
    try:
        verify_result = _run_verify_stage(with_judge, llm, config, out)
        _run_calibrate_stage(out)
        _run_audit_stage(verify_result, out)
    except DemoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - t0
    print(f"done in {elapsed:.2f}s", file=out)
    return 0
