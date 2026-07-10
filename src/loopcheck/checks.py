import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loopcheck.mutation import generate_mutants
from loopcheck.runner import run_pytest


@dataclass
class CheckResult:
    name: str
    score: float
    detail: str


def _has_assertion(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "raises":
                return True
            if isinstance(func, ast.Name) and func.id == "raises":
                return True
    return False


def check_assertions(test_code: str) -> CheckResult:
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        return CheckResult("assertions", 0.0, f"unparseable test code: {e}")
    tests = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]
    if not tests:
        return CheckResult("assertions", 0.0, "no test functions found")
    with_assert = [t for t in tests if _has_assertion(t)]
    score = len(with_assert) / len(tests)
    missing = [t.name for t in tests if t not in with_assert]
    detail = "all tests assert" if not missing else f"tests without assertions: {missing}"
    return CheckResult("assertions", score, detail)


def check_no_target_mock(test_code: str, module_name: str) -> CheckResult:
    patterns = [
        rf"patch\(\s*['\"]{re.escape(module_name)}(?:\.|['\"])",
        rf"monkeypatch\.\w+\(\s*{re.escape(module_name)}\b",
        rf"monkeypatch\.\w+\(\s*['\"]{re.escape(module_name)}[.'\"]",
    ]
    for p in patterns:
        if re.search(p, test_code):
            return CheckResult(
                "no_target_mock", 0.0, f"test code patches target module {module_name}"
            )
    return CheckResult("no_target_mock", 1.0, "target module is not mocked")


def check_tests_pass(
    test_code: str, module_source: str, module_name: str, timeout_s: int = 60
) -> CheckResult:
    r = run_pytest(test_code, module_source, module_name, timeout_s)
    if r.all_passed:
        return CheckResult("tests_pass", 1.0, f"{r.passed} tests pass")
    return CheckResult(
        "tests_pass", 0.0,
        f"collected={r.collected} passed={r.passed} failed={r.failed} "
        f"errored={r.errored}\n{r.output[-1500:]}",
    )


def check_coverage(
    test_code: str, module_source: str, module_name: str, timeout_s: int = 60
) -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        module_file = f"{module_name}.py"
        (tmp_path / module_file).write_text(module_source)
        (tmp_path / "test_generated.py").write_text(test_code)
        try:
            subprocess.run(
                [sys.executable, "-m", "coverage", "run", "--branch",
                 f"--include={module_file}", "-m", "pytest", "test_generated.py", "-q",
                 "-p", "no:cacheprovider"],
                cwd=tmp, capture_output=True, text=True, timeout=timeout_s,
            )
            subprocess.run(
                [sys.executable, "-m", "coverage", "json", "-o", "cov.json"],
                cwd=tmp, capture_output=True, text=True, timeout=30,
            )
            data = json.loads((tmp_path / "cov.json").read_text())
            pct = data["files"][module_file]["summary"]["percent_covered"] / 100.0
            return CheckResult("coverage", pct, f"branch coverage {pct:.0%}")
        except Exception as e:  # coverage failure must not crash verification
            return CheckResult("coverage", 0.0, f"coverage failed: {e}")


def check_mutation(
    test_code: str, module_source: str, module_name: str,
    max_mutants: int = 20, timeout_s: int = 60,
) -> CheckResult:
    mutants = generate_mutants(module_source, max_mutants)
    if not mutants:
        return CheckResult("mutation", 0.5, "no mutants generated")
    survivors = []
    for m in mutants:
        r = run_pytest(test_code, m.source, module_name, timeout_s)
        if r.all_passed:  # mutant survived: tests did not notice the broken code
            survivors.append(m.description)
    kill_rate = 1 - len(survivors) / len(mutants)
    detail = (
        f"killed {len(mutants) - len(survivors)}/{len(mutants)} mutants"
        + (f"; survivors: {survivors}" if survivors else "")
    )
    return CheckResult("mutation", kill_rate, detail)
