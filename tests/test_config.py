from pathlib import Path

from loopcheck.config import Config


def test_defaults():
    c = Config()
    assert c.accept_threshold == 0.85
    assert c.escalate_threshold == 0.50
    assert c.max_iterations == 5
    assert c.agent_model == "claude-sonnet-4-6"
    assert c.db_path == Path("loopcheck.db")
    assert abs(sum(c.weights.values()) - 1.0) < 1e-9


def test_audit_key_env(monkeypatch):
    monkeypatch.setenv("LOOPCHECK_AUDIT_KEY", "sekrit")
    assert Config().audit_key() == b"sekrit"


def test_audit_key_fallback(monkeypatch, capsys):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    assert Config().audit_key() == b"loopcheck-dev-key"
    assert "warning" in capsys.readouterr().err.lower()
