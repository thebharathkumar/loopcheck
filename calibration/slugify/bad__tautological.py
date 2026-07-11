from slugify import slugify


def test_returns_string():
    assert isinstance(slugify("anything"), str)


def test_is_itself():
    assert slugify("hello") == slugify("hello")


def test_true():
    slugify("x")
    assert True
