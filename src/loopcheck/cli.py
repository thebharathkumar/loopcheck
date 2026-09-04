import argparse
import difflib
import os
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
    if args.max_iterations is not None:
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
    from loopcheck.calibrate import run_calibration, save_report

    config = Config(db_path=Path(args.db))
    llm = None
    if not args.no_judge:
        from loopcheck.llm import AnthropicLLM

        llm = AnthropicLLM()
    report = run_calibration(config, llm)
    save_report(report, Path("calibration/report.json"))
    print(f"n={report.n}  precision={report.precision:.2f}  "
          f"recall={report.recall:.2f}  f1={report.f1:.2f}")
    for r in report.rows:
        flag = "OK " if r["label"] == r["predicted"] else "MISS"
        print(f"  {flag} {r['file']}: label={r['label']} conf={r['confidence']}")
    print("report written to calibration/report.json")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    ).returncode


def _cmd_demo(args: argparse.Namespace) -> int:
    from loopcheck.demo import run_demo

    llm = None
    if args.with_judge:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "error: --with-judge requires ANTHROPIC_API_KEY to be set "
                "(no key found in environment)",
                file=sys.stderr,
            )
            return 1
        from loopcheck.llm import AnthropicLLM

        llm = AnthropicLLM(timeout=15.0, max_retries=0)
    return run_demo(with_judge=args.with_judge, llm=llm)


_COMMANDS = ("run", "audit", "calibrate", "dashboard", "demo")


def _resolve_typo(argv: list[str]) -> list[str]:
    """Corrects a misspelled top-level command so a live demo survives a typo."""
    if not argv or argv[0] in _COMMANDS or argv[0].startswith("-"):
        return argv
    match = difflib.get_close_matches(argv[0], _COMMANDS, n=1, cutoff=0.6)
    if not match:
        return argv
    print(f"(interpreting '{argv[0]}' as '{match[0]}')", file=sys.stderr)
    return [match[0], *argv[1:]]


def main(argv: list[str] | None = None) -> int:
    argv = _resolve_typo(sys.argv[1:] if argv is None else list(argv))
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

    p_demo = sub.add_parser(
        "demo",
        help="run the full pipeline live: verify -> calibrate -> audit",
        description=(
            "Runs verify -> calibrate -> audit end to end and prints one short "
            "block per stage.\n\nTwo modes:\n"
            "  (default)     deterministic only. No network calls, no API calls,\n"
            "                no ANTHROPIC_API_KEY needed. Runs in under 2 seconds.\n"
            "                This is the safe default for a live demo.\n"
            "  --with-judge  also makes a live call to the LLM judge. Requires\n"
            "                ANTHROPIC_API_KEY. Fails immediately with a single\n"
            "                clear error if the key is missing or the call fails\n"
            "                -- no hang, no silent retry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_demo.add_argument(
        "--with-judge",
        action="store_true",
        help="also call the live LLM judge (needs ANTHROPIC_API_KEY; fails fast if missing/erroring)",
    )
    p_demo.set_defaults(fn=_cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
