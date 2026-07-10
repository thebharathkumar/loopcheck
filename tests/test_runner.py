from loopcheck.runner import run_pytest

MODULE = "def add(a, b):\n    return a + b\n"


def test_passing_tests():
    r = run_pytest("from mymod import add\ndef test_add():\n    assert add(1, 2) == 3\n", MODULE, "mymod")
    assert r.all_passed and r.passed == 1 and r.failed == 0


def test_failing_tests():
    r = run_pytest("from mymod import add\ndef test_add():\n    assert add(1, 2) == 4\n", MODULE, "mymod")
    assert not r.all_passed and r.failed == 1


def test_no_tests_collected():
    r = run_pytest("x = 1\n", MODULE, "mymod")
    assert r.collected == 0 and not r.all_passed


def test_broken_test_file_errors():
    r = run_pytest("import nonexistent_pkg_xyz\n", MODULE, "mymod")
    assert r.errored and not r.all_passed


def test_timeout():
    r = run_pytest(
        "import time\ndef test_slow():\n    time.sleep(30)\n", MODULE, "mymod", timeout_s=3
    )
    assert r.errored and "timeout" in r.output.lower()
