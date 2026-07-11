from loopcheck.checks import check_coverage, check_mutation, check_tests_pass

MODULE = """
def classify(x):
    if x < 0:
        return "negative"
    if x == 0:
        return "zero"
    return "positive"
"""

STRONG_TESTS = """
from mymod import classify

def test_negative():
    assert classify(-1) == "negative"

def test_zero():
    assert classify(0) == "zero"

def test_positive():
    assert classify(5) == "positive"
"""

WEAK_TESTS = """
from mymod import classify

def test_runs():
    classify(1)
    assert True
"""


def test_tests_pass():
    assert check_tests_pass(STRONG_TESTS, MODULE, "mymod").score == 1.0


def test_tests_fail():
    failing = STRONG_TESTS.replace('"negative"', '"positive"', 1)
    assert check_tests_pass(failing, MODULE, "mymod").score == 0.0


def test_coverage_high_vs_low():
    high = check_coverage(STRONG_TESTS, MODULE, "mymod").score
    low = check_coverage(WEAK_TESTS, MODULE, "mymod").score
    assert high > 0.9 and low < high


def test_mutation_strong_tests_kill_more():
    strong = check_mutation(STRONG_TESTS, MODULE, "mymod", max_mutants=10)
    weak = check_mutation(WEAK_TESTS, MODULE, "mymod", max_mutants=10)
    assert strong.score > weak.score
    assert weak.score < 0.5


def test_mutation_no_mutants():
    r = check_mutation("def test_x():\n    assert True\n", "X = 'const'\n", "mymod")
    assert r.score == 0.5 and "no mutants" in r.detail
