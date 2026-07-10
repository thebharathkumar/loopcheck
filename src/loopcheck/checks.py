import ast
import re
from dataclasses import dataclass


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
