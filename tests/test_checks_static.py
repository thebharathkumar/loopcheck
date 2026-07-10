from loopcheck.checks import check_assertions, check_no_target_mock

GOOD = """
import pytest
from mymod import add

def test_add():
    assert add(1, 2) == 3

def test_add_error():
    with pytest.raises(TypeError):
        add(1, None)
"""

NO_ASSERTS = """
from mymod import add

def test_add():
    add(1, 2)

def test_other():
    x = add(0, 0)
"""

MOCKED = """
from unittest.mock import patch

def test_add():
    with patch("mymod.add", return_value=3):
        from mymod import add
        assert add(1, 2) == 3
"""


def test_assertions_full_score():
    assert check_assertions(GOOD).score == 1.0


def test_assertions_zero():
    assert check_assertions(NO_ASSERTS).score == 0.0


def test_assertions_no_tests_or_broken():
    assert check_assertions("x = 1").score == 0.0
    assert check_assertions("def broken(:").score == 0.0


def test_mock_detected():
    r = check_no_target_mock(MOCKED, "mymod")
    assert r.score == 0.0 and "mymod" in r.detail


def test_no_mock_clean():
    assert check_no_target_mock(GOOD, "mymod").score == 1.0


def test_mock_of_other_module_ok():
    assert check_no_target_mock(MOCKED, "othermod").score == 1.0


def test_mock_of_bare_module_detected():
    code = 'from unittest.mock import patch\n\ndef test_x():\n    with patch("mymod"):\n        assert True\n'
    assert check_no_target_mock(code, "mymod").score == 0.0


def test_async_test_functions_counted():
    code = "import pytest\nfrom mymod import add\n\nasync def test_async_add():\n    assert add(1, 2) == 3\n"
    assert check_assertions(code).score == 1.0
