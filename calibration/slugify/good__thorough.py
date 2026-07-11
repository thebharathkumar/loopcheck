import pytest
from slugify import slugify


def test_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_unicode_normalized():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_empty_and_symbols():
    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_truncation_strips_hyphen():
    assert slugify("aa bb cc", max_length=5) == "aa-bb"


def test_bad_max_length():
    with pytest.raises(ValueError):
        slugify("x", max_length=0)
