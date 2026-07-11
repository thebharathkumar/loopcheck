from slugify import slugify


def test_keeps_case():
    # wrong: the spec says output is always lowercase
    assert slugify("Hello") == "Hello"


def test_underscore_separator():
    # wrong: the spec says runs of non-alphanumerics become hyphens
    assert slugify("a b") == "a_b"


def test_trailing_hyphen_kept():
    # wrong: the spec says truncation never leaves a trailing hyphen
    assert slugify("aa bb", max_length=3) == "aa-"
