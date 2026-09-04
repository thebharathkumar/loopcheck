from loopcheck.audit import append_record
from loopcheck.cli import main
from loopcheck.db import connect


def _seed(db_path, key=b"loopcheck-dev-key"):
    conn = connect(db_path)
    append_record(conn, key, "runA", 1, {"confidence": 0.6, "decision": "retry"})
    append_record(conn, key, "runA", 2, {"confidence": 0.9, "decision": "accept"})
    conn.close()


def test_audit_verify_ok(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    db = str(tmp_path / "lc.db")
    _seed(db)
    assert main(["audit", "verify", "--db", db]) == 0
    assert "chain OK (2 records)" in capsys.readouterr().out


def test_audit_verify_broken(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    db = str(tmp_path / "lc.db")
    _seed(db)
    conn = connect(db)
    conn.execute("UPDATE audit_log SET payload='{}' WHERE seq=1")
    conn.commit()
    assert main(["audit", "verify", "--db", db]) == 1
    assert "chain BROKEN at seq 1" in capsys.readouterr().out


def test_audit_show(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    db = str(tmp_path / "lc.db")
    _seed(db)
    assert main(["audit", "show", "runA", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "retry" in out and "accept" in out


def test_unknown_command_errors():
    try:
        main(["bogus"])
        raised = False
    except SystemExit as e:
        raised = e.code != 0
    assert raised


def test_demo_runs_end_to_end(capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "[1/3] verify" in out and "[2/3] calibrate" in out and "[3/3] audit" in out


def test_demo_typo_is_corrected(capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    assert main(["dem"]) == 0
    captured = capsys.readouterr()
    assert "interpreting 'dem' as 'demo'" in captured.err
    assert "[3/3] audit" in captured.out


def test_demo_with_judge_without_key_fails_immediately(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["demo", "--with-judge"]) == 1
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err
    assert captured.out == ""  # nothing printed before the fast failure
