import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    target TEXT,
    status TEXT,
    started_ts TEXT,
    finished_ts TEXT,
    iterations INTEGER,
    final_confidence REAL,
    cost_usd REAL
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    payload TEXT,
    prev_hmac TEXT,
    hmac TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT,
    trace_id TEXT,
    parent_id TEXT,
    name TEXT,
    start_ns INTEGER,
    end_ns INTEGER,
    attrs TEXT,
    run_id TEXT
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn
