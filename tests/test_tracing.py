import json

from loopcheck.db import connect
from loopcheck.tracing import init_tracing


def test_spans_land_in_sqlite():
    conn = connect(":memory:")
    tracer = init_tracing(conn)
    with tracer.start_as_current_span("iteration", attributes={"loopcheck.run_id": "r1"}):
        with tracer.start_as_current_span("check:mutation", attributes={"score": 0.8}):
            pass
    rows = conn.execute("SELECT * FROM spans ORDER BY start_ns").fetchall()
    assert [r["name"] for r in rows] == ["check:mutation", "iteration"] or [
        r["name"] for r in rows
    ] == ["iteration", "check:mutation"]
    by_name = {r["name"]: r for r in rows}
    assert by_name["iteration"]["run_id"] == "r1"
    assert by_name["check:mutation"]["parent_id"] == by_name["iteration"]["span_id"]
    assert json.loads(by_name["check:mutation"]["attrs"])["score"] == 0.8
    assert by_name["iteration"]["trace_id"] == by_name["check:mutation"]["trace_id"]
