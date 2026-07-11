import argparse
import subprocess
import sys
import uuid
from pathlib import Path

from loopcheck.audit import show_run, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect


def _cmd_run(args: argparse.Namespace) -> int:
    from loopcheck.llm import AnthropicLLM
    from loopcheck.loop import run_loop
    from loopcheck.target import load_target

    config = Config(db_path=Path(args.db))
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    target = load_target(Path(args.target))
    run_id = args.resume or uuid.uuid4().hex[:12]
    state = run_loop(
        target, config, AnthropicLLM(), run_id, resume=bool(args.resume)
    )
    conf = state["confidence_history"][-1] if state["confidence_history"] else float("nan")
    print(f"run_id      {run_id}")
    print(f"decision    {state['decision']}")
    print(f"iterations  {state['iteration']}")
    print(f"confidence  {conf:.3f}")
    print(f"cost_usd    {state['total_cost_usd']:.4f}")
    return 0 if state["decision"] == "accept" else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    key = Config().audit_key()
    if args.audit_cmd == "verify":
        ok, bad_seq = verify_chain(conn, key)
        n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        if ok:
            print(f"chain OK ({n} records)")
            return 0
        print(f"chain BROKEN at seq {bad_seq}")
        return 1
    for rec in show_run(conn, args.run_id):
        print(
            f"iter {rec.get('iteration')}  confidence={rec.get('confidence')}  "
            f"decision={rec.get('decision')}"
        )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    print("calibration not yet implemented")
    return 2


def _cmd_dashboard(args: argparse.Namespace) -> int:
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loopcheck")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the verify-first test-writing loop")
    p_run.add_argument("--target", required=True, help="target directory (module.py + SPEC.md)")
    p_run.add_argument("--resume", default=None, help="run_id to resume")
    p_run.add_argument("--db", default="loopcheck.db")
    p_run.add_argument("--max-iterations", type=int, default=None)
    p_run.set_defaults(fn=_cmd_run)

    p_audit = sub.add_parser("audit", help="verify or inspect the audit chain")
    audit_sub = p_audit.add_subparsers(dest="audit_cmd", required=True)
    p_verify = audit_sub.add_parser("verify")
    p_verify.add_argument("--db", default="loopcheck.db")
    p_verify.set_defaults(fn=_cmd_audit)
    p_show = audit_sub.add_parser("show")
    p_show.add_argument("run_id")
    p_show.add_argument("--db", default="loopcheck.db")
    p_show.set_defaults(fn=_cmd_audit)

    p_cal = sub.add_parser("calibrate", help="evaluate the verifier on the labeled set")
    p_cal.add_argument("--db", default="loopcheck.db")
    p_cal.add_argument("--no-judge", action="store_true")
    p_cal.set_defaults(fn=_cmd_calibrate)

    p_dash = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p_dash.set_defaults(fn=_cmd_dashboard)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
