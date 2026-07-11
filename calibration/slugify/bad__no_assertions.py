from slugify import slugify


def test_basic():
    slugify("Hello, World!")


def test_unicode():
    slugify("Crème Brûlée")


def test_empty():
    slugify("")
