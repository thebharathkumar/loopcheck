import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestRunResult:
    collected: int
    passed: int
    failed: int
    errored: bool
    output: str

    @property
    def all_passed(self) -> bool:
        return self.collected > 0 and self.failed == 0 and not self.errored


def _count(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def run_pytest(
    test_code: str, module_source: str, module_name: str, timeout_s: int = 60
) -> TestRunResult:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / f"{module_name}.py").write_text(module_source)
        (tmp_path / "test_generated.py").write_text(test_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_generated.py", "-q", "--tb=line",
                 "-p", "no:cacheprovider"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return TestRunResult(0, 0, 0, errored=True, output="timeout expired")
        out = proc.stdout + proc.stderr
        passed = _count(r"(\d+) passed", out)
        failed = _count(r"(\d+) failed", out)
        errors = _count(r"(\d+) error", out)
        # exit code 2 = interrupted (e.g. collection error), 3 = internal, 4 = usage
        errored = errors > 0 or proc.returncode in (2, 3, 4)
        return TestRunResult(passed + failed, passed, failed, errored, out[-4000:])
