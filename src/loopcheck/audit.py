import hmac
import json
import sqlite3
from hashlib import sha256

GENESIS = "0" * 64


def _mac(key: bytes, prev_hmac: str, payload_json: str) -> str:
    return hmac.new(key, prev_hmac.encode() + payload_json.encode(), sha256).hexdigest()


def append_record(
    conn: sqlite3.Connection, key: bytes, run_id: str, iteration: int, payload: dict
) -> str:
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    row = conn.execute("SELECT hmac FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row["hmac"] if row else GENESIS
    mac = _mac(key, prev, payload_json)
    conn.execute(
        "INSERT INTO audit_log (run_id, iteration, payload, prev_hmac, hmac) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, iteration, payload_json, prev, mac),
    )
    conn.commit()
    return mac


def verify_chain(conn: sqlite3.Connection, key: bytes) -> tuple[bool, int | None]:
    prev = GENESIS
    for row in conn.execute("SELECT * FROM audit_log ORDER BY seq"):
        expected = _mac(key, prev, row["payload"])
        if row["prev_hmac"] != prev or row["hmac"] != expected:
            return False, row["seq"]
        prev = row["hmac"]
    return True, None


def show_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    return [
        json.loads(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM audit_log WHERE run_id = ? ORDER BY seq", (run_id,)
        )
    ]
