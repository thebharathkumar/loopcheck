from pathlib import Path

from loopcheck.target import load_target

TARGETS = Path(__file__).parent.parent / "targets"


def test_load_target():
    t = load_target(TARGETS / "slugify")
    assert t.name == "slugify"
    assert t.module_name == "slugify"
    assert "def slugify" in t.source
    assert "URL-safe slug" in t.spec
