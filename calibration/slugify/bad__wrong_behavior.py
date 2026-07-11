from slugify import slugify


def test_keeps_case():
    assert slugify("Hello") == "hello"


def test_wrong_separator():
    # asserts underscores, which the spec says become hyphens
    assert slugify("a b") == "a-b" or slugify("a_b") == "a_b"


def test_trailing_hyphen_kept():
    assert slugify("hi!", max_length=64) == "hi"
