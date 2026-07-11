from loopcheck.audit import append_record, show_run, verify_chain
from loopcheck.db import connect

KEY = b"test-key"


def _conn():
    return connect(":memory:")


def test_append_and_verify():
    conn = _conn()
    append_record(conn, KEY, "run1", 1, {"decision": "retry", "confidence": 0.6})
    append_record(conn, KEY, "run1", 2, {"decision": "accept", "confidence": 0.9})
    assert verify_chain(conn, KEY) == (True, None)


def test_tamper_payload_detected_at_record():
    conn = _conn()
    for i in range(3):
        append_record(conn, KEY, "run1", i, {"i": i})
    conn.execute("UPDATE audit_log SET payload = '{\"i\":99}' WHERE seq = 2")
    ok, bad_seq = verify_chain(conn, KEY)
    assert not ok and bad_seq == 2


def test_tamper_breaks_chain_even_with_recomputed_hmac_elsewhere():
    conn = _conn()
    append_record(conn, KEY, "run1", 1, {"a": 1})
    append_record(conn, KEY, "run1", 2, {"a": 2})
    # deleting a middle record breaks the prev_hmac linkage of the next one
    conn.execute("DELETE FROM audit_log WHERE seq = 1")
    ok, bad_seq = verify_chain(conn, KEY)
    assert not ok and bad_seq == 2


def test_wrong_key_fails():
    conn = _conn()
    append_record(conn, KEY, "run1", 1, {"a": 1})
    ok, _ = verify_chain(conn, b"other-key")
    assert not ok


def test_show_run_orders_and_decodes():
    conn = _conn()
    append_record(conn, KEY, "runA", 1, {"d": "retry"})
    append_record(conn, KEY, "runB", 1, {"d": "accept"})
    append_record(conn, KEY, "runA", 2, {"d": "accept"})
    rows = show_run(conn, "runA")
    assert [r["d"] for r in rows] == ["retry", "accept"]
