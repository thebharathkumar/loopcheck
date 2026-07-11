import json
from pathlib import Path

import streamlit as st

from loopcheck.audit import show_run, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect

st.set_page_config(page_title="loopcheck", layout="wide")
st.title("loopcheck — should you trust that it finished?")

db_path = st.sidebar.text_input("Database", "loopcheck.db")
if not Path(db_path).exists():
    st.warning(f"No database at {db_path}. Run `loopcheck run --target targets/slugify` first.")
    st.stop()
conn = connect(db_path)

runs_tab, calib_tab, audit_tab = st.tabs(["Runs", "Calibration", "Audit chain"])

with runs_tab:
    runs = conn.execute("SELECT * FROM runs ORDER BY started_ts DESC").fetchall()
    if not runs:
        st.info("No runs yet.")
    for run in runs:
        with st.expander(
            f"{run['run_id']} — {run['target']} — {run['status']} "
            f"({run['iterations']} iterations, ${run['cost_usd'] or 0:.4f})"
        ):
            records = show_run(conn, run["run_id"])
            if records:
                st.line_chart(
                    {"confidence": [r["confidence"] for r in records]}, height=200
                )
                for r in records:
                    st.markdown(
                        f"**iteration {r['iteration']}** — confidence "
                        f"`{r['confidence']}` → **{r['decision']}**"
                    )
                    st.json(r["checks"], expanded=False)
            spans = conn.execute(
                "SELECT name, (end_ns - start_ns) / 1e6 AS ms, attrs FROM spans "
                "WHERE run_id = ? OR trace_id IN "
                "(SELECT trace_id FROM spans WHERE run_id = ?) ORDER BY start_ns",
                (run["run_id"], run["run_id"]),
            ).fetchall()
            if spans:
                st.caption("Trace")
                st.dataframe(
                    [{"span": s["name"], "duration_ms": round(s["ms"], 1)} for s in spans],
                    use_container_width=True,
                )

with calib_tab:
    report_path = Path("calibration/report.json")
    if not report_path.exists():
        st.info("Run `loopcheck calibrate` to produce calibration/report.json.")
    else:
        rep = json.loads(report_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{rep['precision']:.2f}")
        c2.metric("Recall", f"{rep['recall']:.2f}")
        c3.metric("F1", f"{rep['f1']:.2f}")
        c4.metric("Labeled files", rep["n"])
        st.subheader("Reliability")
        valid = [b for b in rep["bins"] if b["count"]]
        st.bar_chart(
            {f"{b['lo']:.1f}–{b['hi']:.1f}": b["frac_good"] for b in valid}
        )
        st.caption("Fraction of files labeled good, per confidence bin. "
                   "A calibrated verifier trends upward.")
        st.subheader("Per-file results")
        st.dataframe(rep["rows"], use_container_width=True)

with audit_tab:
    ok, bad_seq = verify_chain(conn, Config().audit_key())
    n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    if ok:
        st.success(f"HMAC chain intact — {n} records verified")
    else:
        st.error(f"HMAC chain BROKEN at seq {bad_seq}")
    st.dataframe(
        [dict(r) for r in conn.execute(
            "SELECT seq, run_id, iteration, substr(hmac, 1, 16) AS hmac_prefix "
            "FROM audit_log ORDER BY seq"
        )],
        use_container_width=True,
    )
