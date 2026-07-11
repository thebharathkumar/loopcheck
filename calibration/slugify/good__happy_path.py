from slugify import slugify


def test_lowercases():
    assert slugify("ABC") == "abc"


def test_hyphenates_runs():
    assert slugify("a  --  b") == "a-b"


def test_strips_edges():
    assert slugify("--hello--") == "hello"


def test_truncates():
    assert len(slugify("word " * 50)) <= 64
