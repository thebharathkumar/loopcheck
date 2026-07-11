# loopcheck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build loopcheck — a verifier-first agent loop that writes pytest tests for target modules, scores its own output with a confidence-scored verifier (mutation testing + sanity checks + LLM judge), and reports the verifier's own precision/recall against hand-labeled data. Full OTel tracing, HMAC-chained audit log, Streamlit dashboard.

**Architecture:** LangGraph 3-node StateGraph (generate → verify → decide) with SQLite checkpointing for crash-resume. All logic in plain tested Python; graph nodes are thin wrappers. Verifier combines objective signals (AST mutation-testing kill rate, pytest pass, assertion scan, mock scan, branch coverage) with a Claude judge into a confidence ∈ [0,1] driving accept/retry/escalate.

**Tech Stack:** Python 3.12, uv, LangGraph + langgraph-checkpoint-sqlite, anthropic SDK, pytest + coverage.py, OpenTelemetry SDK, SQLite, Streamlit, ruff.

## Global Constraints

- Python `>=3.12`, managed with `uv`. All commands run via `uv run`.
- Package layout: `src/loopcheck/`. CLI entry point: `loopcheck = "loopcheck.cli:main"`.
- Models (config defaults, always overridable): agent `claude-sonnet-4-6`, judge `claude-haiku-4-5-20251001`.
- Thresholds (config defaults): accept ≥ `0.85`, escalate < `0.50`, max iterations `5`.
- Default DB file: `loopcheck.db` in CWD. Audit key env var: `LOOPCHECK_AUDIT_KEY` (dev fallback `"loopcheck-dev-key"` with printed warning).
- TDD every task: failing test → minimal implementation → pass → commit. Run `uv run ruff check src tests` before every commit.
- No test may call the real Anthropic API. Use `FakeLLM` everywhere in tests.
- pytest is configured with `testpaths = ["tests"]` so `targets/` and `calibration/` are never collected.
- Commit messages: conventional commits (`feat:`, `test:`, `chore:`), each ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

```
pyproject.toml
src/loopcheck/
  __init__.py
  config.py      # Config dataclass (thresholds, weights, models, paths)
  llm.py         # LLM protocol, LLMResponse, AnthropicLLM, FakeLLM, pricing
  runner.py      # subprocess pytest runner (TestRunResult, run_pytest)
  mutation.py    # AST mutation engine (Mutant, generate_mutants)
  checks.py      # CheckResult + static/dynamic checks incl. mutation kill rate
  judge.py       # LLM judge (JudgeResult, judge_tests)
  target.py      # Target dataclass + load_target
  verifier.py    # Verdict, compute_confidence, decide, synthesize_feedback, verify
  db.py          # SQLite schema + connect()
  audit.py       # HMAC chain append/verify/show
  tracing.py     # OTel init + SqliteSpanExporter
  loop.py        # LoopState, graph nodes, build_graph, run_loop
  calibrate.py   # calibration over labeled set → precision/recall/F1 + reliability
  cli.py         # argparse CLI: run, calibrate, audit, dashboard
targets/{slugify,pricing,ratelimit}/{module.py,SPEC.md}
calibration/{slugify,pricing,ratelimit}/{good,bad}__*.py
scripts/gen_flawed.py
dashboard/app.py
tests/           # unit + integration tests (FakeLLM only)
```

---

### Task 1: Project scaffolding + Config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/loopcheck/__init__.py`, `src/loopcheck/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `loopcheck.config.Config` dataclass — fields: `accept_threshold: float = 0.85`, `escalate_threshold: float = 0.50`, `max_iterations: int = 5`, `max_mutants: int = 20`, `test_timeout_s: int = 60`, `agent_model: str = "claude-sonnet-4-6"`, `judge_model: str = "claude-haiku-4-5-20251001"`, `db_path: Path = Path("loopcheck.db")`, `otlp_endpoint: str | None = None`, `weights: dict[str, float]` defaulting to `{"mutation": 0.45, "tests_pass": 0.20, "assertions": 0.10, "no_target_mock": 0.05, "coverage": 0.05, "judge": 0.15}`. Also `Config.audit_key() -> bytes` reading env `LOOPCHECK_AUDIT_KEY`, falling back to `b"loopcheck-dev-key"` and printing a warning to stderr.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "loopcheck"
version = "0.1.0"
description = "Verifier-first agent loop: measures whether you should trust that the agent finished"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40",
    "langgraph>=0.2",
    "langgraph-checkpoint-sqlite>=2.0",
    "opentelemetry-sdk>=1.27",
    "opentelemetry-exporter-otlp-proto-http>=1.27",
    "coverage>=7.6",
    "pytest>=8.0",
    "streamlit>=1.38",
]

[project.scripts]
loopcheck = "loopcheck.cli:main"

[dependency-groups]
dev = ["ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/loopcheck"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

`.gitignore`:

```
__pycache__/
*.egg-info/
.venv/
loopcheck.db
checkpoints.db
*.db-journal
.coverage
cov.json
calibration/report.json
```

- [ ] **Step 2: Write the failing test** — `tests/test_config.py`:

```python
from pathlib import Path

from loopcheck.config import Config


def test_defaults():
    c = Config()
    assert c.accept_threshold == 0.85
    assert c.escalate_threshold == 0.50
    assert c.max_iterations == 5
    assert c.agent_model == "claude-sonnet-4-6"
    assert c.db_path == Path("loopcheck.db")
    assert abs(sum(c.weights.values()) - 1.0) < 1e-9


def test_audit_key_env(monkeypatch):
    monkeypatch.setenv("LOOPCHECK_AUDIT_KEY", "sekrit")
    assert Config().audit_key() == b"sekrit"


def test_audit_key_fallback(monkeypatch, capsys):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    assert Config().audit_key() == b"loopcheck-dev-key"
    assert "warning" in capsys.readouterr().err.lower()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL / collection error (`ModuleNotFoundError: loopcheck.config`)

- [ ] **Step 4: Write minimal implementation** — `src/loopcheck/__init__.py` (empty) and `src/loopcheck/config.py`:

```python
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _default_weights() -> dict[str, float]:
    return {
        "mutation": 0.45,
        "tests_pass": 0.20,
        "assertions": 0.10,
        "no_target_mock": 0.05,
        "coverage": 0.05,
        "judge": 0.15,
    }


@dataclass
class Config:
    accept_threshold: float = 0.85
    escalate_threshold: float = 0.50
    max_iterations: int = 5
    max_mutants: int = 20
    test_timeout_s: int = 60
    agent_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-haiku-4-5-20251001"
    db_path: Path = Path("loopcheck.db")
    otlp_endpoint: str | None = None
    weights: dict[str, float] = field(default_factory=_default_weights)

    def audit_key(self) -> bytes:
        key = os.environ.get("LOOPCHECK_AUDIT_KEY")
        if key:
            return key.encode()
        print(
            "warning: LOOPCHECK_AUDIT_KEY not set, using insecure dev key",
            file=sys.stderr,
        )
        return b"loopcheck-dev-key"
```

- [ ] **Step 5: Run tests, then lint**

Run: `uv run pytest tests/test_config.py -v` → PASS. `uv run ruff check src tests` → clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore uv.lock src tests
git commit -m "feat: project scaffolding and Config"
```

---

### Task 2: Demo target modules + SPECs

**Files:**
- Create: `targets/slugify/module.py`, `targets/slugify/SPEC.md`, `targets/pricing/module.py`, `targets/pricing/SPEC.md`, `targets/ratelimit/module.py`, `targets/ratelimit/SPEC.md`, `src/loopcheck/target.py`
- Test: `tests/test_targets.py`, `tests/test_target_loading.py`

**Interfaces:**
- Produces: `loopcheck.target.Target` dataclass — fields `name: str`, `module_name: str`, `source: str`, `spec: str`; and `load_target(path: Path) -> Target` where `path` is a directory containing `module.py` and `SPEC.md`; `module_name == path.name`.

- [ ] **Step 1: Write failing reference tests for the three targets** — `tests/test_targets.py`:

```python
import pytest

from targets.pricing.module import price_order
from targets.ratelimit.module import TokenBucket
from targets.slugify.module import slugify


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_unicode():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_slugify_empty_and_symbols():
    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_slugify_truncates_without_trailing_hyphen():
    assert slugify("aa bb cc", max_length=5) == "aa-bb"
    with pytest.raises(ValueError):
        slugify("x", max_length=0)


def test_pricing_tiers():
    assert price_order(5, 10.0) == 50.00          # 0% tier
    assert price_order(10, 10.0) == 95.00         # 5% tier boundary
    assert price_order(50, 10.0) == 450.00        # 10% tier boundary
    assert price_order(100, 10.0) == 850.00       # 15% tier boundary


def test_pricing_vip_and_cap():
    assert price_order(5, 10.0, "vip") == 47.50   # 0+5 = 5%
    assert price_order(100, 10.0, "vip") == 800.00  # 15+5 = 20% (cap)


def test_pricing_rounding_half_up():
    assert price_order(1, 0.125, "vip") == 0.12   # 0.125*0.95=0.11875 -> 0.12


def test_pricing_errors():
    with pytest.raises(ValueError):
        price_order(-1, 10.0)
    with pytest.raises(ValueError):
        price_order(1, 10.0, "gold")


def test_bucket_burst_then_deny():
    t = [0.0]
    b = TokenBucket(capacity=3, refill_rate=1.0, clock=lambda: t[0])
    assert [b.allow(), b.allow(), b.allow(), b.allow()] == [True, True, True, False]


def test_bucket_refills_capped():
    t = [0.0]
    b = TokenBucket(capacity=2, refill_rate=1.0, clock=lambda: t[0])
    b.allow()
    b.allow()
    t[0] = 10.0  # refill far beyond capacity
    assert b.allow() and b.allow() and not b.allow()


def test_bucket_oversized_request():
    b = TokenBucket(capacity=2, refill_rate=1.0)
    with pytest.raises(ValueError):
        b.allow(3)
```

Imports resolve because Step 3 creates empty `__init__.py` files in `targets/` and each subdirectory, making `targets` a package importable from the repo root (pytest's rootdir).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_targets.py -v`
Expected: FAIL (`ModuleNotFoundError: targets`)

- [ ] **Step 3: Implement the three targets**

`targets/slugify/module.py`:

```python
import re
import unicodedata


def slugify(text: str, max_length: int = 64) -> str:
    """Convert text to a URL-safe slug. See SPEC.md."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-")
    return cleaned
```

`targets/slugify/SPEC.md`:

```markdown
# slugify(text, max_length=64) -> str

Converts arbitrary text into a URL-safe slug.

- Unicode is NFKD-normalized and non-ASCII characters dropped ("Crème" → "creme"; emoji vanish).
- Result is lowercase; every run of non-alphanumeric characters becomes a single hyphen.
- No leading or trailing hyphens, even after truncation.
- Truncated to `max_length` characters; a hyphen left dangling by truncation is removed.
- Empty or all-symbol input returns "".
- `max_length < 1` raises ValueError.
```

`targets/pricing/module.py`:

```python
from decimal import ROUND_HALF_UP, Decimal

_TIERS = [(100, Decimal("0.15")), (50, Decimal("0.10")), (10, Decimal("0.05")), (0, Decimal("0"))]
_VIP_BONUS = Decimal("0.05")
_MAX_DISCOUNT = Decimal("0.20")


def price_order(quantity: int, unit_price: float, customer_type: str = "standard") -> float:
    """Total price after tiered quantity discount. See SPEC.md."""
    if quantity < 0:
        raise ValueError("quantity must be >= 0")
    if customer_type not in ("standard", "vip"):
        raise ValueError(f"unknown customer_type: {customer_type}")
    discount = next(d for threshold, d in _TIERS if quantity >= threshold)
    if customer_type == "vip":
        discount = min(discount + _VIP_BONUS, _MAX_DISCOUNT)
    total = Decimal(quantity) * Decimal(str(unit_price)) * (1 - discount)
    return float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
```

`targets/pricing/SPEC.md`:

```markdown
# price_order(quantity, unit_price, customer_type="standard") -> float

Total order price after tiered quantity discounts, rounded half-up to cents.

- Tiers by quantity: 0–9 → 0%, 10–49 → 5%, 50–99 → 10%, 100+ → 15%. Boundaries inclusive.
- customer_type "vip" adds 5 percentage points, capped at 20% total discount.
- Rounding is decimal half-up (0.11875 → 0.12), not banker's rounding.
- quantity < 0 raises ValueError; customer_type other than "standard"/"vip" raises ValueError.
- quantity 0 returns 0.0.
```

`targets/ratelimit/module.py`:

```python
import time
from collections.abc import Callable


class TokenBucket:
    """Token-bucket rate limiter with injectable clock. See SPEC.md."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1 or refill_rate <= 0:
            raise ValueError("capacity must be >= 1 and refill_rate > 0")
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def allow(self, tokens: int = 1) -> bool:
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        if tokens > self.capacity:
            raise ValueError("request exceeds bucket capacity")
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_rate)
        self._last = now
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False
```

`targets/ratelimit/SPEC.md`:

```markdown
# TokenBucket(capacity, refill_rate, clock=time.monotonic)

Token-bucket rate limiter. `allow(tokens=1) -> bool` consumes tokens if available.

- Starts full (capacity tokens). Allows bursts up to capacity, then denies.
- Refills at refill_rate tokens/second based on elapsed clock time; never exceeds capacity.
- Denied calls still advance the refill clock (no token loss).
- `clock` is injectable for deterministic tests.
- capacity < 1 or refill_rate <= 0 raises ValueError at construction.
- allow(tokens) with tokens < 1 raises ValueError; tokens > capacity raises ValueError.
```

Create empty `__init__.py` in `targets/`, `targets/slugify/`, `targets/pricing/`, `targets/ratelimit/`.

- [ ] **Step 4: Run reference tests** — `uv run pytest tests/test_targets.py -v` → all PASS.

- [ ] **Step 5: Write failing loader test** — `tests/test_target_loading.py`:

```python
from pathlib import Path

from loopcheck.target import load_target

TARGETS = Path(__file__).parent.parent / "targets"


def test_load_target():
    t = load_target(TARGETS / "slugify")
    assert t.name == "slugify"
    assert t.module_name == "slugify"
    assert "def slugify" in t.source
    assert "URL-safe slug" in t.spec
```

- [ ] **Step 6: Implement loader** — `src/loopcheck/target.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Target:
    name: str
    module_name: str
    source: str
    spec: str


def load_target(path: Path) -> Target:
    name = path.name
    return Target(
        name=name,
        module_name=name,
        source=(path / "module.py").read_text(),
        spec=(path / "SPEC.md").read_text(),
    )
```

- [ ] **Step 7: Run all tests + lint + commit**

Run: `uv run pytest -v` → PASS; `uv run ruff check src tests targets` → clean.

```bash
git add targets src/loopcheck/target.py tests
git commit -m "feat: demo target modules with SPECs and Target loader"
```

---

### Task 3: Subprocess pytest runner

**Files:**
- Create: `src/loopcheck/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `loopcheck.runner.TestRunResult` dataclass — `collected: int`, `passed: int`, `failed: int`, `errored: bool`, `output: str`, property `all_passed: bool` (True iff `collected > 0 and failed == 0 and not errored`); and `run_pytest(test_code: str, module_source: str, module_name: str, timeout_s: int = 60) -> TestRunResult`. Writes `{module_name}.py` + `test_generated.py` into a temp dir, runs `sys.executable -m pytest test_generated.py -q --tb=line -p no:cacheprovider` with `cwd=tmpdir`, parses the summary line.

- [ ] **Step 1: Write the failing test** — `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_runner.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/runner.py`:

```python
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
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_runner.py -v` → PASS (timeout test takes ~3s).

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/runner.py tests/test_runner.py
git commit -m "feat: subprocess pytest runner with timeout"
```

---

### Task 4: AST mutation engine

**Files:**
- Create: `src/loopcheck/mutation.py`
- Test: `tests/test_mutation.py`

**Interfaces:**
- Produces: `loopcheck.mutation.Mutant` dataclass — `operator: str`, `description: str`, `source: str`; and `generate_mutants(source: str, max_mutants: int = 20) -> list[Mutant]`. Operators (exact names): `flip_comparison` (`<`↔`<=`, `>`↔`>=`, `==`↔`!=`), `off_by_one` (int constant n → n+1; bools excluded), `negate_condition` (`if X:` → `if not X:`), `swap_operands` (BinOp with `-`, `/`, `//`, `%`: swap left/right), `delete_branch` (If body → `pass`), `flip_bool` (True↔False). Each mutant applies exactly one mutation; results are round-robin interleaved across operators then truncated to `max_mutants`.

- [ ] **Step 1: Write the failing test** — `tests/test_mutation.py`:

```python
import ast

from loopcheck.mutation import generate_mutants

SRC = '''
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def is_adult(age):
    return age >= 18

LIMIT = 10
ENABLED = True
'''


def _ops(mutants):
    return {m.operator for m in mutants}


def test_generates_mutants_for_each_operator():
    mutants = generate_mutants(SRC, max_mutants=100)
    assert {"flip_comparison", "off_by_one", "negate_condition",
            "delete_branch", "flip_bool"} <= _ops(mutants)


def test_each_mutant_differs_and_compiles():
    for m in generate_mutants(SRC, max_mutants=100):
        assert m.source != SRC
        ast.parse(m.source)  # must be valid python


def test_flip_comparison_actually_flips():
    mutants = [m for m in generate_mutants(SRC, max_mutants=100) if m.operator == "flip_comparison"]
    assert any("x <= lo" in m.source for m in mutants)


def test_max_mutants_cap():
    assert len(generate_mutants(SRC, max_mutants=3)) == 3


def test_swap_operands():
    src = "def sub(a, b):\n    return a - b\n"
    mutants = [m for m in generate_mutants(src) if m.operator == "swap_operands"]
    assert len(mutants) == 1 and "b - a" in mutants[0].source


def test_no_mutable_sites():
    assert generate_mutants("x = 'hello'\n") == []
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_mutation.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/mutation.py`:

```python
import ast
import copy
from dataclasses import dataclass

_CMP_SWAP: dict[type, type] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_SWAPPABLE_BINOPS = (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod)

OPERATORS = [
    "flip_comparison", "off_by_one", "negate_condition",
    "swap_operands", "delete_branch", "flip_bool",
]


@dataclass
class Mutant:
    operator: str
    description: str
    source: str


class _Mutator(ast.NodeTransformer):
    """Applies exactly one mutation: the target_idx-th applicable site for `op`."""

    def __init__(self, op: str, target_idx: int) -> None:
        self.op = op
        self.target_idx = target_idx
        self.count = 0
        self.applied_at: int | None = None  # line number, None if not applied

    def _hit(self, node: ast.AST) -> bool:
        hit = self.count == self.target_idx
        self.count += 1
        if hit:
            self.applied_at = getattr(node, "lineno", 0)
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.op == "flip_comparison":
            for i, cmp_op in enumerate(node.ops):
                if type(cmp_op) in _CMP_SWAP and self._hit(node):
                    node.ops[i] = _CMP_SWAP[type(cmp_op)]()
                    break
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.op == "off_by_one" and type(node.value) is int and self._hit(node):
            return ast.copy_location(ast.Constant(node.value + 1), node)
        if self.op == "flip_bool" and type(node.value) is bool and self._hit(node):
            return ast.copy_location(ast.Constant(not node.value), node)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if self.op == "negate_condition" and self._hit(node):
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif self.op == "delete_branch" and self._hit(node):
            node.body = [ast.Pass()]
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            self.op == "swap_operands"
            and isinstance(node.op, _SWAPPABLE_BINOPS)
            and self._hit(node)
        ):
            node.left, node.right = node.right, node.left
        return node


def _mutants_for_op(tree: ast.AST, source: str, op: str) -> list[Mutant]:
    mutants = []
    idx = 0
    while True:
        mutator = _Mutator(op, idx)
        mutated = mutator.visit(copy.deepcopy(tree))
        if mutator.applied_at is None:
            break
        new_source = ast.unparse(ast.fix_missing_locations(mutated))
        if new_source != ast.unparse(tree):
            mutants.append(
                Mutant(op, f"{op} at line {mutator.applied_at}", new_source)
            )
        idx += 1
    return mutants


def generate_mutants(source: str, max_mutants: int = 20) -> list[Mutant]:
    tree = ast.parse(source)
    per_op = [_mutants_for_op(tree, source, op) for op in OPERATORS]
    interleaved: list[Mutant] = []
    i = 0
    while any(per_op) and len(interleaved) < max_mutants:
        for lst in per_op:
            if i < len(lst) and len(interleaved) < max_mutants:
                interleaved.append(lst[i])
        i += 1
        if all(i >= len(lst) for lst in per_op):
            break
    return interleaved
```

Note: `visit_Constant` — bools in Python are ints, so check `type(node.value) is int` (excludes bools) for off_by_one and `type(node.value) is bool` for flip_bool.

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_mutation.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/mutation.py tests/test_mutation.py
git commit -m "feat: AST mutation engine with six operators"
```

---

### Task 5: Static sanity checks

**Files:**
- Create: `src/loopcheck/checks.py`
- Test: `tests/test_checks_static.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `loopcheck.checks.CheckResult` dataclass — `name: str`, `score: float` (0..1), `detail: str`; `check_assertions(test_code: str) -> CheckResult` (name `"assertions"`; score = fraction of `test_*` functions containing an `assert` statement or a `pytest.raises` context; 0.0 if no test functions or unparseable); `check_no_target_mock(test_code: str, module_name: str) -> CheckResult` (name `"no_target_mock"`; score 1.0 if the module is never patched/monkeypatched, 0.0 otherwise).

- [ ] **Step 1: Write the failing test** — `tests/test_checks_static.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_checks_static.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/checks.py`:

```python
import ast
import re
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    score: float
    detail: str


def _has_assertion(fn: ast.FunctionDef) -> bool:
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
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
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
        rf"patch\(\s*['\"]{re.escape(module_name)}[.'\"]",
        rf"monkeypatch\.\w+\(\s*{re.escape(module_name)}\b",
        rf"monkeypatch\.\w+\(\s*['\"]{re.escape(module_name)}[.'\"]",
    ]
    for p in patterns:
        if re.search(p, test_code):
            return CheckResult(
                "no_target_mock", 0.0, f"test code patches target module {module_name}"
            )
    return CheckResult("no_target_mock", 1.0, "target module is not mocked")
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_checks_static.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/checks.py tests/test_checks_static.py
git commit -m "feat: static sanity checks (assertions, target mocking)"
```

---

### Task 6: Dynamic checks — tests-pass, coverage, mutation kill rate

**Files:**
- Modify: `src/loopcheck/checks.py` (append functions)
- Test: `tests/test_checks_dynamic.py`

**Interfaces:**
- Consumes: `run_pytest(test_code, module_source, module_name, timeout_s) -> TestRunResult` (Task 3), `generate_mutants(source, max_mutants) -> list[Mutant]` (Task 4), `CheckResult` (Task 5).
- Produces (appended to `loopcheck.checks`): `check_tests_pass(test_code: str, module_source: str, module_name: str, timeout_s: int = 60) -> CheckResult` (name `"tests_pass"`, score 1.0 if `all_passed` else 0.0); `check_coverage(test_code, module_source, module_name, timeout_s=60) -> CheckResult` (name `"coverage"`, score = branch-coverage fraction of the module file, 0.0 on any failure); `check_mutation(test_code, module_source, module_name, max_mutants=20, timeout_s=60) -> CheckResult` (name `"mutation"`, score = kill rate; a mutant is killed if the run is not `all_passed`; score 0.5 with detail `"no mutants generated"` when the engine yields none; detail lists surviving mutant descriptions).

- [ ] **Step 1: Write the failing test** — `tests/test_checks_dynamic.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_checks_dynamic.py -v` → ImportError.

- [ ] **Step 3: Implement** — append to `src/loopcheck/checks.py`:

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from loopcheck.mutation import generate_mutants
from loopcheck.runner import run_pytest


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
```

Move the new imports (`json`, `subprocess`, `sys`, `tempfile`, `Path`, `generate_mutants`, `run_pytest`) to the top of `checks.py` with the existing imports.

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_checks_dynamic.py -v` → PASS (this test spawns ~25 pytest subprocesses; expect ~30–60s).

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/checks.py tests/test_checks_dynamic.py
git commit -m "feat: dynamic checks - tests-pass, branch coverage, mutation kill rate"
```

---

### Task 7: LLM layer (protocol, Anthropic client, FakeLLM, pricing)

**Files:**
- Create: `src/loopcheck/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `loopcheck.llm.LLMResponse` dataclass — `text: str`, `input_tokens: int`, `output_tokens: int`, `cost_usd: float`; protocol `LLM` with method `complete(system: str, user: str, model: str, max_tokens: int = 4096) -> LLMResponse`; `AnthropicLLM` (lazy `anthropic.Anthropic()` client, uses SDK retries); `FakeLLM(responses: list[str])` popping queued responses (raises `IndexError` if exhausted, records `calls: list[tuple[str, str]]` of (system, user)); `cost_usd(model: str, input_tokens: int, output_tokens: int) -> float` using per-MTok prices `{"claude-sonnet-4-6": (3.0, 15.0), "claude-haiku-4-5-20251001": (1.0, 5.0)}`, unknown models → 0.0.

- [ ] **Step 1: Write the failing test** — `tests/test_llm.py`:

```python
import pytest

from loopcheck.llm import FakeLLM, cost_usd


def test_cost_known_model():
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert cost_usd("claude-haiku-4-5-20251001", 2_000_000, 0) == pytest.approx(2.0)


def test_cost_unknown_model():
    assert cost_usd("gpt-x", 1000, 1000) == 0.0


def test_fake_llm_pops_and_records():
    fake = FakeLLM(["first", "second"])
    r1 = fake.complete("sys", "hello", "any-model")
    assert r1.text == "first" and r1.cost_usd == 0.0
    assert fake.complete("sys", "again", "any-model").text == "second"
    assert fake.calls == [("sys", "hello"), ("sys", "again")]
    with pytest.raises(IndexError):
        fake.complete("sys", "third", "any-model")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_llm.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/llm.py`:

```python
from dataclasses import dataclass, field
from typing import Protocol

# USD per million tokens: (input, output)
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return input_tokens / 1e6 * inp + output_tokens / 1e6 * out


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLM(Protocol):
    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 4096
    ) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self) -> None:
        self._client = None

    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 4096
    ) -> LLMResponse:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # SDK handles retries
        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            cost_usd=cost_usd(model, msg.usage.input_tokens, msg.usage.output_tokens),
        )


@dataclass
class FakeLLM:
    responses: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 4096
    ) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(self.responses.pop(0), 10, 10, 0.0)
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_llm.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/llm.py tests/test_llm.py
git commit -m "feat: LLM layer with Anthropic client, FakeLLM, and pricing"
```

---

### Task 8: LLM judge

**Files:**
- Create: `src/loopcheck/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `LLM`, `LLMResponse` (Task 7).
- Produces: `loopcheck.judge.JudgeResult` dataclass — `score: float` (mean of criteria), `criteria: dict[str, float]`, `rationale: str`, `cost_usd: float`; `judge_tests(llm: LLM, model: str, spec: str, module_source: str, test_code: str) -> JudgeResult`. Criteria keys (exact): `intent`, `edge_cases`, `assertion_quality`, `independence`. Raises `ValueError` if no JSON object can be extracted from the response.

- [ ] **Step 1: Write the failing test** — `tests/test_judge.py`:

```python
import json

import pytest

from loopcheck.judge import judge_tests
from loopcheck.llm import FakeLLM

GOOD_JSON = json.dumps({
    "criteria": {"intent": 1.0, "edge_cases": 0.5, "assertion_quality": 1.0, "independence": 0.5},
    "rationale": "covers happy path, misses unicode edge cases",
})


def test_judge_parses_scores():
    fake = FakeLLM([GOOD_JSON])
    r = judge_tests(fake, "m", "SPEC", "def f(): pass", "def test_f(): assert True")
    assert r.score == pytest.approx(0.75)
    assert r.criteria["edge_cases"] == 0.5
    assert "unicode" in r.rationale


def test_judge_extracts_json_from_chatter():
    fake = FakeLLM(["Here is my assessment:\n" + GOOD_JSON + "\nHope that helps!"])
    assert judge_tests(fake, "m", "s", "src", "tests").score == pytest.approx(0.75)


def test_judge_prompt_contains_inputs():
    fake = FakeLLM([GOOD_JSON])
    judge_tests(fake, "m", "THE_SPEC", "THE_SOURCE", "THE_TESTS")
    _, user = fake.calls[0]
    assert "THE_SPEC" in user and "THE_SOURCE" in user and "THE_TESTS" in user


def test_judge_rejects_garbage():
    with pytest.raises(ValueError):
        judge_tests(FakeLLM(["no json here"]), "m", "s", "src", "tests")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_judge.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/judge.py`:

```python
import json
from dataclasses import dataclass

from loopcheck.llm import LLM

CRITERIA = ["intent", "edge_cases", "assertion_quality", "independence"]

_SYSTEM = """You are a rigorous test-quality judge. You evaluate whether a pytest test file \
genuinely verifies a module's INTENDED behavior per its spec — not merely whether tests pass.
Score each criterion from 0.0 to 1.0:
- intent: do the tests assert the behaviors the spec describes?
- edge_cases: are the spec's edge cases (boundaries, errors, empty inputs) covered?
- assertion_quality: are assertions specific (exact values) rather than trivial or tautological?
- independence: do tests exercise the real module (no mocking it away, no re-implementing it)?
Respond with ONLY a JSON object: {"criteria": {"intent": x, "edge_cases": x, \
"assertion_quality": x, "independence": x}, "rationale": "<2-4 sentences>"}"""

_USER_TEMPLATE = """## Module spec
{spec}

## Module source
```python
{module_source}
```

## Test file under evaluation
```python
{test_code}
```"""


@dataclass
class JudgeResult:
    score: float
    criteria: dict[str, float]
    rationale: str
    cost_usd: float


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def judge_tests(
    llm: LLM, model: str, spec: str, module_source: str, test_code: str
) -> JudgeResult:
    user = _USER_TEMPLATE.format(spec=spec, module_source=module_source, test_code=test_code)
    resp = llm.complete(_SYSTEM, user, model, max_tokens=1024)
    data = _extract_json(resp.text)
    criteria = {k: float(data["criteria"].get(k, 0.0)) for k in CRITERIA}
    return JudgeResult(
        score=sum(criteria.values()) / len(criteria),
        criteria=criteria,
        rationale=str(data.get("rationale", "")),
        cost_usd=resp.cost_usd,
    )
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_judge.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/judge.py tests/test_judge.py
git commit -m "feat: rubric-scored LLM judge"
```

---

### Task 9: Verifier — confidence combiner, decision, feedback

**Files:**
- Create: `src/loopcheck/verifier.py`
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: `CheckResult`, `check_tests_pass`, `check_assertions`, `check_no_target_mock`, `check_coverage`, `check_mutation` (Tasks 5–6), `JudgeResult`, `judge_tests` (Task 8), `Target` (Task 2), `Config` (Task 1), `LLM` (Task 7).
- Produces: `loopcheck.verifier.Verdict` dataclass — `confidence: float`, `decision: str` (`"accept" | "retry" | "escalate"`), `checks: list[CheckResult]`, `judge: JudgeResult | None`, `feedback: str`, `cost_usd: float`; `compute_confidence(checks: list[CheckResult], judge: JudgeResult | None, weights: dict[str, float]) -> float` (weighted mean over present signals; weights renormalized when judge absent); `decide(confidence: float, history: list[float], config: Config) -> str` (accept if ≥ accept_threshold; escalate if < escalate_threshold; escalate on stagnation: `len(history) >= 2 and confidence <= history[-1] <= history[-2]`; else retry); `synthesize_feedback(checks, judge) -> str` (concatenates detail of every check scoring < 1.0 plus judge rationale if judge score < 0.9); `verify(test_code: str, target: Target, llm: LLM | None, config: Config, history: list[float]) -> Verdict` — orchestrates all checks; judge skipped when `llm is None`.

- [ ] **Step 1: Write the failing test** — `tests/test_verifier.py`:

```python
import json

import pytest

from loopcheck.checks import CheckResult
from loopcheck.config import Config
from loopcheck.judge import JudgeResult
from loopcheck.llm import FakeLLM
from loopcheck.target import Target
from loopcheck.verifier import compute_confidence, decide, synthesize_feedback, verify

CFG = Config()


def _checks(mutation=1.0, tests_pass=1.0, assertions=1.0, no_target_mock=1.0, coverage=1.0):
    return [
        CheckResult("mutation", mutation, "m"),
        CheckResult("tests_pass", tests_pass, "t"),
        CheckResult("assertions", assertions, "a"),
        CheckResult("no_target_mock", no_target_mock, "n"),
        CheckResult("coverage", coverage, "c"),
    ]


def _judge(score):
    return JudgeResult(score, {"intent": score}, "rationale text", 0.0)


def test_confidence_perfect():
    assert compute_confidence(_checks(), _judge(1.0), CFG.weights) == pytest.approx(1.0)


def test_confidence_weighted():
    conf = compute_confidence(_checks(mutation=0.0), _judge(1.0), CFG.weights)
    assert conf == pytest.approx(0.55)  # lost the 0.45 mutation weight


def test_confidence_without_judge_renormalizes():
    conf = compute_confidence(_checks(), None, CFG.weights)
    assert conf == pytest.approx(1.0)


def test_decide_thresholds():
    assert decide(0.90, [], CFG) == "accept"
    assert decide(0.60, [], CFG) == "retry"
    assert decide(0.40, [], CFG) == "escalate"


def test_decide_stagnation_escalates():
    assert decide(0.60, [0.70, 0.65], CFG) == "escalate"   # two non-improvements
    assert decide(0.70, [0.50, 0.65], CFG) == "retry"      # still improving


def test_feedback_mentions_failures_only():
    checks = _checks(mutation=0.4)
    checks[0] = CheckResult("mutation", 0.4, "survivors: [off_by_one at line 3]")
    fb = synthesize_feedback(checks, _judge(0.5))
    assert "off_by_one" in fb and "rationale text" in fb
    assert "no_target_mock" not in fb


def test_verify_end_to_end_with_fake_judge():
    target = Target(
        name="mymod", module_name="mymod",
        source="def double(x):\n    if x < 0:\n        return -2 * x\n    return 2 * x\n",
        spec="double(x): returns 2*x for x>=0, -2*x for x<0",
    )
    tests = (
        "from mymod import double\n"
        "def test_pos():\n    assert double(3) == 6\n"
        "def test_neg():\n    assert double(-3) == 6\n"
        "def test_zero():\n    assert double(0) == 0\n"
    )
    judge_json = json.dumps({
        "criteria": {"intent": 1.0, "edge_cases": 1.0, "assertion_quality": 1.0,
                     "independence": 1.0},
        "rationale": "solid",
    })
    v = verify(tests, target, FakeLLM([judge_json]), CFG, history=[])
    assert 0.0 < v.confidence <= 1.0
    assert v.decision in ("accept", "retry", "escalate")
    assert {c.name for c in v.checks} == {
        "mutation", "tests_pass", "assertions", "no_target_mock", "coverage"
    }


def test_verify_without_llm_skips_judge():
    target = Target("mymod", "mymod", "def one():\n    return 1\n", "one() returns 1")
    v = verify("from mymod import one\ndef test_one():\n    assert one() == 1\n",
               target, None, CFG, history=[])
    assert v.judge is None and v.confidence > 0.0
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_verifier.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/verifier.py`:

```python
from dataclasses import dataclass

from loopcheck.checks import (
    CheckResult,
    check_assertions,
    check_coverage,
    check_mutation,
    check_no_target_mock,
    check_tests_pass,
)
from loopcheck.config import Config
from loopcheck.judge import JudgeResult, judge_tests
from loopcheck.llm import LLM
from loopcheck.target import Target


@dataclass
class Verdict:
    confidence: float
    decision: str
    checks: list[CheckResult]
    judge: JudgeResult | None
    feedback: str
    cost_usd: float


def compute_confidence(
    checks: list[CheckResult], judge: JudgeResult | None, weights: dict[str, float]
) -> float:
    scores = {c.name: c.score for c in checks}
    if judge is not None:
        scores["judge"] = judge.score
    present = {k: w for k, w in weights.items() if k in scores}
    total = sum(present.values())
    if total == 0:
        return 0.0
    return sum(scores[k] * w for k, w in present.items()) / total


def decide(confidence: float, history: list[float], config: Config) -> str:
    if confidence >= config.accept_threshold:
        return "accept"
    if confidence < config.escalate_threshold:
        return "escalate"
    if len(history) >= 2 and confidence <= history[-1] <= history[-2]:
        return "escalate"  # stagnating: two consecutive non-improvements
    return "retry"


def synthesize_feedback(checks: list[CheckResult], judge: JudgeResult | None) -> str:
    parts = [f"[{c.name}] score={c.score:.2f}: {c.detail}" for c in checks if c.score < 1.0]
    if judge is not None and judge.score < 0.9:
        parts.append(f"[judge] score={judge.score:.2f}: {judge.rationale}")
    return "\n".join(parts) if parts else "all checks passed"


def verify(
    test_code: str,
    target: Target,
    llm: LLM | None,
    config: Config,
    history: list[float],
) -> Verdict:
    checks = [
        check_tests_pass(test_code, target.source, target.module_name, config.test_timeout_s),
        check_assertions(test_code),
        check_no_target_mock(test_code, target.module_name),
        check_coverage(test_code, target.source, target.module_name, config.test_timeout_s),
        check_mutation(
            test_code, target.source, target.module_name,
            config.max_mutants, config.test_timeout_s,
        ),
    ]
    judge = None
    if llm is not None:
        judge = judge_tests(llm, config.judge_model, target.spec, target.source, test_code)
    confidence = compute_confidence(checks, judge, config.weights)
    return Verdict(
        confidence=confidence,
        decision=decide(confidence, history, config),
        checks=checks,
        judge=judge,
        feedback=synthesize_feedback(checks, judge),
        cost_usd=judge.cost_usd if judge else 0.0,
    )
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_verifier.py -v` → PASS (spawns subprocesses; ~20–40s).

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/verifier.py tests/test_verifier.py
git commit -m "feat: confidence-scored verifier with decision and feedback synthesis"
```

---

### Task 10: SQLite storage + HMAC audit chain

**Files:**
- Create: `src/loopcheck/db.py`, `src/loopcheck/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces: `loopcheck.db.connect(path: Path | str) -> sqlite3.Connection` — creates tables if absent: `runs(run_id TEXT PRIMARY KEY, target TEXT, status TEXT, started_ts TEXT, finished_ts TEXT, iterations INTEGER, final_confidence REAL, cost_usd REAL)`; `audit_log(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, iteration INTEGER, payload TEXT, prev_hmac TEXT, hmac TEXT)`; `spans(span_id TEXT, trace_id TEXT, parent_id TEXT, name TEXT, start_ns INTEGER, end_ns INTEGER, attrs TEXT, run_id TEXT)`. Connection uses `row_factory = sqlite3.Row`.
- Produces: `loopcheck.audit.append_record(conn, key: bytes, run_id: str, iteration: int, payload: dict) -> str` (returns hmac hex; payload stored as canonical JSON `json.dumps(payload, sort_keys=True, separators=(",", ":"))`; `hmac_hex = HMAC_SHA256(key, prev_hmac_hex.encode() + payload_json.encode())`; genesis `prev_hmac = "0" * 64`); `verify_chain(conn, key: bytes) -> tuple[bool, int | None]` (walks all rows ordered by seq; returns `(True, None)` or `(False, first_bad_seq)`); `show_run(conn, run_id: str) -> list[dict]` (decoded payloads in seq order).

- [ ] **Step 1: Write the failing test** — `tests/test_audit.py`:

```python
from loopcheck.audit import append_record, show_run, verify_chain
from loopcheck.db import connect

KEY = b"test-key"


def _conn():
    return connect(":memory:")


def test_append_and_verify():
    conn = _conn()
    append_record(conn, KEY, "run1", 1, {"decision": "retry", "confidence": 0.6})
    append_record(conn, KEY, "run1", 2, {"decision": "accept", "confidence": 0.9})
    assert verify_chain(conn, KEY) == (True, None)


def test_tamper_payload_detected_at_record():
    conn = _conn()
    for i in range(3):
        append_record(conn, KEY, "run1", i, {"i": i})
    conn.execute("UPDATE audit_log SET payload = '{\"i\":99}' WHERE seq = 2")
    ok, bad_seq = verify_chain(conn, KEY)
    assert not ok and bad_seq == 2


def test_tamper_breaks_chain_even_with_recomputed_hmac_elsewhere():
    conn = _conn()
    append_record(conn, KEY, "run1", 1, {"a": 1})
    append_record(conn, KEY, "run1", 2, {"a": 2})
    # deleting a middle record breaks the prev_hmac linkage of the next one
    conn.execute("DELETE FROM audit_log WHERE seq = 1")
    ok, bad_seq = verify_chain(conn, KEY)
    assert not ok and bad_seq == 2


def test_wrong_key_fails():
    conn = _conn()
    append_record(conn, KEY, "run1", 1, {"a": 1})
    ok, _ = verify_chain(conn, b"other-key")
    assert not ok


def test_show_run_orders_and_decodes():
    conn = _conn()
    append_record(conn, KEY, "runA", 1, {"d": "retry"})
    append_record(conn, KEY, "runB", 1, {"d": "accept"})
    append_record(conn, KEY, "runA", 2, {"d": "accept"})
    rows = show_run(conn, "runA")
    assert [r["d"] for r in rows] == ["retry", "accept"]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_audit.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/db.py`:

```python
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    target TEXT,
    status TEXT,
    started_ts TEXT,
    finished_ts TEXT,
    iterations INTEGER,
    final_confidence REAL,
    cost_usd REAL
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    payload TEXT,
    prev_hmac TEXT,
    hmac TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT,
    trace_id TEXT,
    parent_id TEXT,
    name TEXT,
    start_ns INTEGER,
    end_ns INTEGER,
    attrs TEXT,
    run_id TEXT
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn
```

`src/loopcheck/audit.py`:

```python
import hmac
import json
import sqlite3
from hashlib import sha256

GENESIS = "0" * 64


def _mac(key: bytes, prev_hmac: str, payload_json: str) -> str:
    return hmac.new(key, prev_hmac.encode() + payload_json.encode(), sha256).hexdigest()


def append_record(
    conn: sqlite3.Connection, key: bytes, run_id: str, iteration: int, payload: dict
) -> str:
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    row = conn.execute("SELECT hmac FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row["hmac"] if row else GENESIS
    mac = _mac(key, prev, payload_json)
    conn.execute(
        "INSERT INTO audit_log (run_id, iteration, payload, prev_hmac, hmac) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, iteration, payload_json, prev, mac),
    )
    conn.commit()
    return mac


def verify_chain(conn: sqlite3.Connection, key: bytes) -> tuple[bool, int | None]:
    prev = GENESIS
    for row in conn.execute("SELECT * FROM audit_log ORDER BY seq"):
        expected = _mac(key, prev, row["payload"])
        if row["prev_hmac"] != prev or row["hmac"] != expected:
            return False, row["seq"]
        prev = row["hmac"]
    return True, None


def show_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    return [
        json.loads(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM audit_log WHERE run_id = ? ORDER BY seq", (run_id,)
        )
    ]
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_audit.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/db.py src/loopcheck/audit.py tests/test_audit.py
git commit -m "feat: SQLite storage and tamper-evident HMAC audit chain"
```

---

### Task 11: OpenTelemetry tracing with SQLite span exporter

**Files:**
- Create: `src/loopcheck/tracing.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Consumes: `connect` (Task 10).
- Produces: `loopcheck.tracing.SqliteSpanExporter(conn: sqlite3.Connection)` — OTel `SpanExporter` writing each span into the `spans` table (`span_id`/`trace_id`/`parent_id` as hex strings, `attrs` as JSON; `run_id` copied from span attribute `"loopcheck.run_id"` if present); `init_tracing(conn, otlp_endpoint: str | None = None) -> opentelemetry.trace.Tracer` — builds a fresh `TracerProvider` (NOT the global one, so tests are isolated), attaches `SimpleSpanProcessor(SqliteSpanExporter(conn))`, plus an OTLP HTTP exporter if `otlp_endpoint` given; returns `provider.get_tracer("loopcheck")`.

- [ ] **Step 1: Write the failing test** — `tests/test_tracing.py`:

```python
import json

from loopcheck.db import connect
from loopcheck.tracing import init_tracing


def test_spans_land_in_sqlite():
    conn = connect(":memory:")
    tracer = init_tracing(conn)
    with tracer.start_as_current_span("iteration", attributes={"loopcheck.run_id": "r1"}) as parent:
        with tracer.start_as_current_span("check:mutation", attributes={"score": 0.8}):
            pass
    rows = conn.execute("SELECT * FROM spans ORDER BY start_ns").fetchall()
    assert [r["name"] for r in rows] == ["check:mutation", "iteration"] or [
        r["name"] for r in rows
    ] == ["iteration", "check:mutation"]
    by_name = {r["name"]: r for r in rows}
    assert by_name["iteration"]["run_id"] == "r1"
    assert by_name["check:mutation"]["parent_id"] == by_name["iteration"]["span_id"]
    assert json.loads(by_name["check:mutation"]["attrs"])["score"] == 0.8
    assert by_name["iteration"]["trace_id"] == by_name["check:mutation"]["trace_id"]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_tracing.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/tracing.py`:

```python
import json
import sqlite3
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class SqliteSpanExporter(SpanExporter):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            ctx = span.get_span_context()
            attrs = dict(span.attributes or {})
            parent_id = f"{span.parent.span_id:016x}" if span.parent else None
            self._conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{ctx.span_id:016x}",
                    f"{ctx.trace_id:032x}",
                    parent_id,
                    span.name,
                    span.start_time,
                    span.end_time,
                    json.dumps(attrs, default=str),
                    attrs.get("loopcheck.run_id"),
                ),
            )
        self._conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def init_tracing(
    conn: sqlite3.Connection, otlp_endpoint: str | None = None
) -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SqliteSpanExporter(conn)))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    return provider.get_tracer("loopcheck")
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_tracing.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/tracing.py tests/test_tracing.py
git commit -m "feat: OTel tracing with SQLite span exporter and optional OTLP"
```

---

### Task 12: LangGraph loop with checkpointing

**Files:**
- Create: `src/loopcheck/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: everything above — `Config`, `Target`/`load_target`, `LLM`/`FakeLLM`, `verify`/`Verdict`, `connect`, `append_record`, `init_tracing`, `cost_usd`.
- Produces: `loopcheck.loop.LoopState` TypedDict — `run_id: str`, `target_name: str`, `test_code: str`, `feedback: str`, `iteration: int`, `confidence_history: list[float]`, `decision: str`, `total_cost_usd: float`; `extract_code_block(text: str) -> str` (first ```python fenced block, else whole text); `run_loop(target: Target, config: Config, llm: LLM, run_id: str, checkpoint_path: str = "checkpoints.db", resume: bool = False) -> LoopState` — builds and invokes the graph, opens `connect(config.db_path)`, initializes tracing, writes `runs` row on start (status `"running"`) and finish (status = final decision or `"unconverged"`), appends one audit record per verification (payload keys: `run_id`, `iteration`, `ts`, `confidence`, `decision`, `checks` (name→score map), `rationale_sha256`), records spans `generate`/`verify` with child spans per check, and sets `decision` in final state. Graph: `generate` → `verify` → conditional: retry & iteration < max → `generate`, else END. Uses `langgraph.checkpoint.sqlite.SqliteSaver` with `thread_id = run_id`; `resume=True` invokes with input `None` to continue from checkpoint.

- [ ] **Step 1: Write the failing test** — `tests/test_loop.py`:

```python
import json

from loopcheck.audit import show_run, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect
from loopcheck.llm import FakeLLM
from loopcheck.loop import extract_code_block, run_loop
from loopcheck.target import Target

TARGET = Target(
    name="mymod", module_name="mymod",
    source="def double(x):\n    if x < 0:\n        return -2 * x\n    return 2 * x\n",
    spec="double(x): 2*x for x>=0, -2*x for x<0. Always non-negative doubling magnitude.",
)

GOOD_TESTS = """```python
from mymod import double

def test_pos():
    assert double(3) == 6

def test_neg():
    assert double(-3) == 6

def test_zero():
    assert double(0) == 0
```"""

BAD_TESTS = """```python
from mymod import double

def test_pos():
    assert double(3) == 6
```"""

PERFECT_JUDGE = json.dumps({
    "criteria": {"intent": 1.0, "edge_cases": 1.0, "assertion_quality": 1.0,
                 "independence": 1.0},
    "rationale": "thorough",
})
HARSH_JUDGE = json.dumps({
    "criteria": {"intent": 0.2, "edge_cases": 0.0, "assertion_quality": 0.1,
                 "independence": 1.0},
    "rationale": "only the positive path is tested",
})
# Expected confidence with BAD_TESTS ≈ 0.61 (retry band), GOOD_TESTS ≈ 0.91 (accept):
# BAD kills ~2/5 mutants (positive-path only), GOOD kills ~4/5 plus perfect judge.


def test_extract_code_block():
    assert extract_code_block(GOOD_TESTS).startswith("from mymod import double")
    assert extract_code_block("plain code") == "plain code"


def test_loop_converges_second_iteration(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db", max_iterations=5)
    llm = FakeLLM([BAD_TESTS, HARSH_JUDGE, GOOD_TESTS, PERFECT_JUDGE])
    state = run_loop(TARGET, cfg, llm, run_id="run-x",
                     checkpoint_path=str(tmp_path / "ckpt.db"))
    assert state["decision"] == "accept"
    assert state["iteration"] == 2
    assert len(state["confidence_history"]) == 2
    assert state["confidence_history"][1] > state["confidence_history"][0]

    conn = connect(cfg.db_path)
    run = conn.execute("SELECT * FROM runs WHERE run_id='run-x'").fetchone()
    assert run["status"] == "accept" and run["iterations"] == 2

    records = show_run(conn, "run-x")
    assert [r["decision"] for r in records] == ["retry", "accept"]
    assert verify_chain(conn, cfg.audit_key()) == (True, None)

    span_names = {r["name"] for r in conn.execute("SELECT name FROM spans")}
    assert {"generate", "verify", "check:mutation"} <= span_names


def test_loop_feedback_reaches_next_generation(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db")
    llm = FakeLLM([BAD_TESTS, HARSH_JUDGE, GOOD_TESTS, PERFECT_JUDGE])
    run_loop(TARGET, cfg, llm, run_id="run-y", checkpoint_path=str(tmp_path / "ckpt.db"))
    second_generate_prompt = llm.calls[2][1]
    assert "only the positive path is tested" in second_generate_prompt  # judge rationale fed back


def test_loop_stops_at_max_iterations(tmp_path):
    cfg = Config(db_path=tmp_path / "lc.db", max_iterations=2)
    llm = FakeLLM([BAD_TESTS, HARSH_JUDGE] * 2)
    state = run_loop(TARGET, cfg, llm, run_id="run-z",
                     checkpoint_path=str(tmp_path / "ckpt.db"))
    assert state["iteration"] == 2
    assert state["decision"] in ("retry", "escalate")
    conn = connect(cfg.db_path)
    run = conn.execute("SELECT * FROM runs WHERE run_id='run-z'").fetchone()
    assert run["status"] in ("unconverged", "escalate")
```

Note on FakeLLM ordering: each iteration consumes two responses — generate (test file) then judge (JSON) — because `verify` calls the judge once per iteration.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_loop.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/loop.py`:

```python
import json
import re
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from loopcheck.audit import append_record
from loopcheck.config import Config
from loopcheck.db import connect
from loopcheck.llm import LLM
from loopcheck.target import Target
from loopcheck.tracing import init_tracing
from loopcheck.verifier import verify


class LoopState(TypedDict):
    run_id: str
    target_name: str
    test_code: str
    feedback: str
    iteration: int
    confidence_history: list[float]
    decision: str
    total_cost_usd: float


_GENERATE_SYSTEM = """You write high-quality pytest test files. You will receive a module's \
spec and source. Write a single self-contained pytest file that verifies the INTENDED behavior \
in the spec, including edge cases and error conditions. Import the module by its module name. \
Use exact-value assertions. Do not mock the module under test. \
Respond with ONLY a fenced python code block."""


def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() + "\n" if m else text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_generate_prompt(target: Target, state: LoopState) -> str:
    prompt = (
        f"## Module name\n{target.module_name}\n\n## Spec\n{target.spec}\n\n"
        f"## Source\n```python\n{target.source}\n```"
    )
    if state["test_code"]:
        prompt += (
            f"\n\n## Your previous attempt\n```python\n{state['test_code']}\n```"
            f"\n\n## Verifier feedback (fix these issues)\n{state['feedback']}"
        )
    return prompt


def run_loop(
    target: Target,
    config: Config,
    llm: LLM,
    run_id: str,
    checkpoint_path: str = "checkpoints.db",
    resume: bool = False,
) -> LoopState:
    conn = connect(config.db_path)
    tracer = init_tracing(conn, config.otlp_endpoint)
    audit_key = config.audit_key()

    def generate(state: LoopState) -> dict:
        with tracer.start_as_current_span(
            "generate",
            attributes={"loopcheck.run_id": run_id, "iteration": state["iteration"] + 1},
        ) as span:
            resp = llm.complete(
                _GENERATE_SYSTEM, _build_generate_prompt(target, state), config.agent_model
            )
            span.set_attribute("cost_usd", resp.cost_usd)
            span.set_attribute("output_tokens", resp.output_tokens)
            return {
                "test_code": extract_code_block(resp.text),
                "iteration": state["iteration"] + 1,
                "total_cost_usd": state["total_cost_usd"] + resp.cost_usd,
            }

    def verify_node(state: LoopState) -> dict:
        with tracer.start_as_current_span(
            "verify",
            attributes={"loopcheck.run_id": run_id, "iteration": state["iteration"]},
        ) as span:
            verdict = None
            history = state["confidence_history"]
            with tracer.start_as_current_span("checks"):
                verdict = verify(state["test_code"], target, llm, config, history)
            for c in verdict.checks:
                with tracer.start_as_current_span(
                    f"check:{c.name}", attributes={"score": c.score, "detail": c.detail[:500]}
                ):
                    pass
            span.set_attribute("confidence", verdict.confidence)
            span.set_attribute("decision", verdict.decision)
            span.set_attribute("cost_usd", verdict.cost_usd)
            rationale = verdict.judge.rationale if verdict.judge else ""
            append_record(
                conn, audit_key, run_id, state["iteration"],
                {
                    "run_id": run_id,
                    "iteration": state["iteration"],
                    "ts": _now(),
                    "confidence": round(verdict.confidence, 4),
                    "decision": verdict.decision,
                    "checks": {c.name: round(c.score, 4) for c in verdict.checks},
                    "rationale_sha256": sha256(rationale.encode()).hexdigest(),
                },
            )
            return {
                "confidence_history": history + [verdict.confidence],
                "decision": verdict.decision,
                "feedback": verdict.feedback,
                "total_cost_usd": state["total_cost_usd"] + verdict.cost_usd,
            }

    def route(state: LoopState) -> str:
        if state["decision"] == "retry" and state["iteration"] < config.max_iterations:
            return "generate"
        return END

    graph = StateGraph(LoopState)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route)

    initial: LoopState = {
        "run_id": run_id,
        "target_name": target.name,
        "test_code": "",
        "feedback": "",
        "iteration": 0,
        "confidence_history": [],
        "decision": "",
        "total_cost_usd": 0.0,
    }
    conn.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, 'running', ?, NULL, 0, NULL, 0)",
        (run_id, target.name, _now()),
    )
    conn.commit()

    with SqliteSaver.from_conn_string(checkpoint_path) as saver:
        compiled = graph.compile(checkpointer=saver)
        thread = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        final = compiled.invoke(None if resume else initial, thread)

    status = final["decision"]
    if status == "retry":  # hit the iteration cap while still retrying
        status = "unconverged"
    conn.execute(
        "UPDATE runs SET status=?, finished_ts=?, iterations=?, final_confidence=?, "
        "cost_usd=? WHERE run_id=?",
        (status, _now(), final["iteration"],
         final["confidence_history"][-1] if final["confidence_history"] else None,
         final["total_cost_usd"], run_id),
    )
    conn.commit()
    conn.close()
    return final  # type: ignore[return-value]
```

Implementation notes for the executor:
- `sqlite3` import in loop.py is unused if you don't reference it — omit it.
- If `SqliteSaver.from_conn_string` in the installed langgraph-checkpoint-sqlite version is not a context manager, use `SqliteSaver(sqlite3.connect(checkpoint_path, check_same_thread=False))` instead — check with `uv run python -c "from langgraph.checkpoint.sqlite import SqliteSaver; help(SqliteSaver.from_conn_string)"`.
- SQLite connections are per-thread sensitive; LangGraph runs nodes on the caller thread with `.invoke`, so a single `conn` is fine.

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_loop.py -v` → PASS (~60–90s: each iteration runs mutation testing in subprocesses).

- [ ] **Step 5: Run the full suite** — `uv run pytest` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/loopcheck/loop.py tests/test_loop.py
git commit -m "feat: LangGraph verify-first loop with checkpointing, tracing, audit"
```

---

### Task 13: CLI

**Files:**
- Create: `src/loopcheck/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `run_loop`, `load_target`, `Config`, `connect`, `verify_chain`, `show_run`, `AnthropicLLM`.
- Produces: `loopcheck.cli.main(argv: list[str] | None = None) -> int`. Subcommands:
  - `loopcheck run --target <dir> [--resume RUN_ID] [--db PATH] [--max-iterations N]` — generates `run_id = uuid4().hex[:12]` (or uses `--resume` value), calls `run_loop`, prints a summary (run id, decision, iterations, final confidence, cost).
  - `loopcheck audit verify [--db PATH]` — prints `chain OK (N records)` and returns 0, or `chain BROKEN at seq X` and returns 1.
  - `loopcheck audit show RUN_ID [--db PATH]` — prints one line per record: iteration, confidence, decision.
  - `loopcheck calibrate [--db PATH] [--no-judge]` — implemented in Task 14; until then registers and prints `calibration not yet implemented`, returns 2.
  - `loopcheck dashboard` — `subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli.py` (drive `main()` directly; audit data seeded via library calls):

```python
from loopcheck.audit import append_record
from loopcheck.cli import main
from loopcheck.db import connect


def _seed(db_path, key=b"loopcheck-dev-key"):
    conn = connect(db_path)
    append_record(conn, key, "runA", 1, {"confidence": 0.6, "decision": "retry"})
    append_record(conn, key, "runA", 2, {"confidence": 0.9, "decision": "accept"})
    conn.close()


def test_audit_verify_ok(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    db = str(tmp_path / "lc.db")
    _seed(db)
    assert main(["audit", "verify", "--db", db]) == 0
    assert "chain OK (2 records)" in capsys.readouterr().out


def test_audit_verify_broken(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    db = str(tmp_path / "lc.db")
    _seed(db)
    conn = connect(db)
    conn.execute("UPDATE audit_log SET payload='{}' WHERE seq=1")
    conn.commit()
    assert main(["audit", "verify", "--db", db]) == 1
    assert "chain BROKEN at seq 1" in capsys.readouterr().out


def test_audit_show(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOPCHECK_AUDIT_KEY", raising=False)
    db = str(tmp_path / "lc.db")
    _seed(db)
    assert main(["audit", "show", "runA", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "retry" in out and "accept" in out


def test_unknown_command_errors():
    try:
        main(["bogus"])
        raised = False
    except SystemExit as e:
        raised = e.code != 0
    assert raised
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_cli.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — `src/loopcheck/cli.py`:

```python
import argparse
import subprocess
import sys
import uuid
from pathlib import Path

from loopcheck.audit import show_run, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect


def _cmd_run(args: argparse.Namespace) -> int:
    from loopcheck.llm import AnthropicLLM
    from loopcheck.loop import run_loop
    from loopcheck.target import load_target

    config = Config(db_path=Path(args.db))
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    target = load_target(Path(args.target))
    run_id = args.resume or uuid.uuid4().hex[:12]
    state = run_loop(
        target, config, AnthropicLLM(), run_id, resume=bool(args.resume)
    )
    conf = state["confidence_history"][-1] if state["confidence_history"] else float("nan")
    print(f"run_id      {run_id}")
    print(f"decision    {state['decision']}")
    print(f"iterations  {state['iteration']}")
    print(f"confidence  {conf:.3f}")
    print(f"cost_usd    {state['total_cost_usd']:.4f}")
    return 0 if state["decision"] == "accept" else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    key = Config().audit_key()
    if args.audit_cmd == "verify":
        ok, bad_seq = verify_chain(conn, key)
        n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        if ok:
            print(f"chain OK ({n} records)")
            return 0
        print(f"chain BROKEN at seq {bad_seq}")
        return 1
    for rec in show_run(conn, args.run_id):
        print(
            f"iter {rec.get('iteration')}  confidence={rec.get('confidence')}  "
            f"decision={rec.get('decision')}"
        )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    print("calibration not yet implemented")
    return 2


def _cmd_dashboard(args: argparse.Namespace) -> int:
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loopcheck")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the verify-first test-writing loop")
    p_run.add_argument("--target", required=True, help="target directory (module.py + SPEC.md)")
    p_run.add_argument("--resume", default=None, help="run_id to resume")
    p_run.add_argument("--db", default="loopcheck.db")
    p_run.add_argument("--max-iterations", type=int, default=None)
    p_run.set_defaults(fn=_cmd_run)

    p_audit = sub.add_parser("audit", help="verify or inspect the audit chain")
    audit_sub = p_audit.add_subparsers(dest="audit_cmd", required=True)
    p_verify = audit_sub.add_parser("verify")
    p_verify.add_argument("--db", default="loopcheck.db")
    p_verify.set_defaults(fn=_cmd_audit)
    p_show = audit_sub.add_parser("show")
    p_show.add_argument("run_id")
    p_show.add_argument("--db", default="loopcheck.db")
    p_show.set_defaults(fn=_cmd_audit)

    p_cal = sub.add_parser("calibrate", help="evaluate the verifier on the labeled set")
    p_cal.add_argument("--db", default="loopcheck.db")
    p_cal.add_argument("--no-judge", action="store_true")
    p_cal.set_defaults(fn=_cmd_calibrate)

    p_dash = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p_dash.set_defaults(fn=_cmd_dashboard)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_cli.py -v` → PASS. Also smoke: `uv run loopcheck audit verify --db /tmp/empty.db` prints `chain OK (0 records)`.

- [ ] **Step 5: Commit**

```bash
git add src/loopcheck/cli.py tests/test_cli.py
git commit -m "feat: CLI - run, audit verify/show, calibrate stub, dashboard"
```

---

### Task 14: Calibration — labeled seed set, generator script, metrics

**Files:**
- Create: `calibration/README.md`, 15 seed files under `calibration/{slugify,pricing,ratelimit}/`, `scripts/gen_flawed.py`, `src/loopcheck/calibrate.py`
- Modify: `src/loopcheck/cli.py` (`_cmd_calibrate`)
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `verify`, `Config`, `load_target`, `AnthropicLLM`/`FakeLLM`.
- Produces: `loopcheck.calibrate.CalibrationReport` dataclass — `precision: float`, `recall: float`, `f1: float`, `n: int`, `rows: list[dict]` (each: `file`, `target`, `label`, `confidence`, `predicted`), `bins: list[dict]` (each: `lo`, `hi`, `count`, `mean_confidence`, `frac_good`); `run_calibration(config: Config, llm: LLM | None, calibration_dir: Path = Path("calibration"), targets_dir: Path = Path("targets")) -> CalibrationReport`; `save_report(report, path: Path)` writing JSON. Labeled-file convention: `calibration/<target>/<label>__<slug>.py` where label ∈ {`good`, `bad`}. Positive class = `good`; prediction = `confidence >= config.accept_threshold`. precision = TP/(TP+FP), recall = TP/(TP+FN) (0.0 when denominator is 0). Bins: 5 equal-width confidence bins [0,0.2)…[0.8,1.0].

- [ ] **Step 1: Create the labeled seed set (15 files).** For each target, 2 good + 3 bad. The bad ones cover flaw archetypes: assertion-free, tautological, wrong-behavior. Full content for slugify (pricing and ratelimit follow the identical pattern against their own SPECs — write real tests, not placeholders):

`calibration/slugify/good__thorough.py`:

```python
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
```

`calibration/slugify/good__happy_path.py`:

```python
from slugify import slugify


def test_lowercases():
    assert slugify("ABC") == "abc"


def test_hyphenates_runs():
    assert slugify("a  --  b") == "a-b"


def test_strips_edges():
    assert slugify("--hello--") == "hello"


def test_truncates():
    assert len(slugify("word " * 50)) <= 64
```

`calibration/slugify/bad__no_assertions.py`:

```python
from slugify import slugify


def test_basic():
    slugify("Hello, World!")


def test_unicode():
    slugify("Crème Brûlée")


def test_empty():
    slugify("")
```

`calibration/slugify/bad__tautological.py`:

```python
from slugify import slugify


def test_returns_string():
    assert isinstance(slugify("anything"), str)


def test_is_itself():
    assert slugify("hello") == slugify("hello")


def test_true():
    slugify("x")
    assert True
```

`calibration/slugify/bad__wrong_behavior.py`:

```python
from slugify import slugify


def test_keeps_case():
    assert slugify("Hello") == "hello"


def test_wrong_separator():
    # asserts underscores, which the spec says become hyphens
    assert slugify("a b") == "a-b" or slugify("a_b") == "a_b"


def test_trailing_hyphen_kept():
    assert slugify("hi!", max_length=64) == "hi"
```

For `calibration/pricing/` write `good__thorough.py` (tier boundaries 9/10/49/50/99/100, vip cap, rounding, both ValueErrors), `good__happy_path.py` (a few exact-value checks), `bad__no_assertions.py`, `bad__tautological.py` (`isinstance(..., float)`, `x == x`), `bad__wrong_behavior.py` (asserts 10% at qty 10, ignores vip cap). For `calibration/ratelimit/` write `good__thorough.py` (burst-then-deny, capped refill, denied-call clock advance, all three ValueErrors — using injected clock), `good__happy_path.py`, `bad__no_assertions.py`, `bad__tautological.py`, `bad__wrong_behavior.py` (asserts bucket starts empty). Import each module by its module name (`from pricing import price_order`, `from ratelimit import TokenBucket`).

`calibration/README.md`:

```markdown
# Calibration set

Hand-labeled test files used to measure the verifier itself (precision/recall of
"accept"). Naming: `<target>/<label>__<slug>.py`, label ∈ {good, bad}.

- good: genuinely verifies the target's SPEC.md behavior
- bad: passes or superficially plausible but does NOT verify intended behavior
  (assertion-free, tautological, wrong-behavior, target-mocked-away)

Seeds are hand-written. Expand with `uv run python scripts/gen_flawed.py <target>`
(uses the Anthropic API), then hand-review every generated file before trusting
the labels — the label is the ground truth being measured against.
```

- [ ] **Step 2: Write scripts/gen_flawed.py** (expansion helper, not used in tests):

```python
"""Generate additional labeled calibration files via the Anthropic API.

Usage: uv run python scripts/gen_flawed.py <target-name> [count]
Writes candidates to calibration/<target>/candidate__*.py for HAND REVIEW.
Rename to good__*.py / bad__*.py only after reviewing.
"""
import sys
from pathlib import Path

from loopcheck.config import Config
from loopcheck.llm import AnthropicLLM
from loopcheck.loop import extract_code_block
from loopcheck.target import load_target

FLAWS = [
    ("subtle_wrong", "tests that look thorough but assert one subtly wrong expected value"),
    ("shallow", "tests that only exercise the single happiest path with weak assertions"),
    ("overmocked", "tests that patch/mock the module under test so nothing real runs"),
]


def main() -> int:
    name = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else len(FLAWS)
    target = load_target(Path("targets") / name)
    llm = AnthropicLLM()
    cfg = Config()
    out_dir = Path("calibration") / name
    for i, (slug, flaw) in enumerate(FLAWS[:count]):
        resp = llm.complete(
            "You write pytest files with a specific deliberate flaw, for evaluating "
            "a test-quality verifier. Respond with ONLY a fenced python code block.",
            f"Module `{name}` spec:\n{target.spec}\n\nSource:\n```python\n{target.source}\n```"
            f"\n\nWrite a pytest file importing `{name}` that has this flaw: {flaw}.",
            cfg.agent_model,
        )
        path = out_dir / f"candidate__{slug}_{i}.py"
        path.write_text(extract_code_block(resp.text))
        print(f"wrote {path} — hand-review and rename to good__/bad__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Write the failing calibration test** — `tests/test_calibrate.py`:

```python
import json
from pathlib import Path

from loopcheck.calibrate import run_calibration, save_report
from loopcheck.config import Config


def test_calibration_no_judge_separates_good_from_bad(tmp_path):
    cfg = Config(max_mutants=8)
    report = run_calibration(cfg, llm=None)
    assert report.n == 15
    good_rows = [r for r in report.rows if r["label"] == "good"]
    bad_rows = [r for r in report.rows if r["label"] == "bad"]
    mean_good = sum(r["confidence"] for r in good_rows) / len(good_rows)
    mean_bad = sum(r["confidence"] for r in bad_rows) / len(bad_rows)
    assert mean_good > mean_bad  # the verifier must at least rank good above bad
    assert 0.0 <= report.precision <= 1.0 and 0.0 <= report.recall <= 1.0
    assert sum(b["count"] for b in report.bins) == report.n

    out = tmp_path / "report.json"
    save_report(report, out)
    data = json.loads(out.read_text())
    assert data["n"] == 15 and len(data["rows"]) == 15
```

- [ ] **Step 4: Run to verify failure** — `uv run pytest tests/test_calibrate.py -v` → ModuleNotFoundError.

- [ ] **Step 5: Implement** — `src/loopcheck/calibrate.py`:

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loopcheck.config import Config
from loopcheck.llm import LLM
from loopcheck.target import load_target
from loopcheck.verifier import verify


@dataclass
class CalibrationReport:
    precision: float
    recall: float
    f1: float
    n: int
    rows: list[dict]
    bins: list[dict]


def run_calibration(
    config: Config,
    llm: LLM | None,
    calibration_dir: Path = Path("calibration"),
    targets_dir: Path = Path("targets"),
) -> CalibrationReport:
    rows = []
    for target_dir in sorted(p for p in calibration_dir.iterdir() if p.is_dir()):
        target = load_target(targets_dir / target_dir.name)
        for f in sorted(target_dir.glob("*__*.py")):
            label = f.name.split("__")[0]
            if label not in ("good", "bad"):
                continue
            verdict = verify(f.read_text(), target, llm, config, history=[])
            rows.append({
                "file": f"{target_dir.name}/{f.name}",
                "target": target_dir.name,
                "label": label,
                "confidence": round(verdict.confidence, 4),
                "predicted": "good" if verdict.confidence >= config.accept_threshold else "bad",
            })
    tp = sum(1 for r in rows if r["label"] == "good" and r["predicted"] == "good")
    fp = sum(1 for r in rows if r["label"] == "bad" and r["predicted"] == "good")
    fn = sum(1 for r in rows if r["label"] == "good" and r["predicted"] == "bad")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bins = []
    for i in range(5):
        lo, hi = i * 0.2, (i + 1) * 0.2
        in_bin = [r for r in rows if lo <= r["confidence"] < hi or (i == 4 and r["confidence"] == 1.0)]
        bins.append({
            "lo": lo,
            "hi": hi,
            "count": len(in_bin),
            "mean_confidence": round(
                sum(r["confidence"] for r in in_bin) / len(in_bin), 4
            ) if in_bin else None,
            "frac_good": round(
                sum(1 for r in in_bin if r["label"] == "good") / len(in_bin), 4
            ) if in_bin else None,
        })
    return CalibrationReport(precision, recall, f1, len(rows), rows, bins)


def save_report(report: CalibrationReport, path: Path) -> None:
    path.write_text(json.dumps(asdict(report), indent=2))
```

Replace `_cmd_calibrate` in `src/loopcheck/cli.py`:

```python
def _cmd_calibrate(args: argparse.Namespace) -> int:
    from loopcheck.calibrate import run_calibration, save_report

    config = Config(db_path=Path(args.db))
    llm = None
    if not args.no_judge:
        from loopcheck.llm import AnthropicLLM

        llm = AnthropicLLM()
    report = run_calibration(config, llm)
    save_report(report, Path("calibration/report.json"))
    print(f"n={report.n}  precision={report.precision:.2f}  "
          f"recall={report.recall:.2f}  f1={report.f1:.2f}")
    for r in report.rows:
        flag = "OK " if r["label"] == r["predicted"] else "MISS"
        print(f"  {flag} {r['file']}: label={r['label']} conf={r['confidence']}")
    print("report written to calibration/report.json")
    return 0
```

- [ ] **Step 6: Run tests + lint** — `uv run pytest tests/test_calibrate.py -v` → PASS (runs the full verifier on 15 files without judge; expect 2–5 min). Then full suite `uv run pytest`.

- [ ] **Step 7: Commit**

```bash
git add calibration scripts src/loopcheck/calibrate.py src/loopcheck/cli.py tests/test_calibrate.py
git commit -m "feat: calibration seed set, generator script, precision/recall report"
```

---

### Task 15: Streamlit dashboard

**Files:**
- Create: `dashboard/app.py`
- Test: manual smoke (Streamlit apps aren't unit-tested here; keep all logic in loopcheck modules)

**Interfaces:**
- Consumes: `connect` (Task 10), `verify_chain`/`show_run` (Task 10), `Config`, `calibration/report.json` (Task 14).

- [ ] **Step 1: Implement** — `dashboard/app.py`:

```python
import json
from pathlib import Path

import streamlit as st

from loopcheck.audit import show_run, verify_chain
from loopcheck.config import Config
from loopcheck.db import connect

st.set_page_config(page_title="loopcheck", layout="wide")
st.title("loopcheck — should you trust that it finished?")

db_path = st.sidebar.text_input("Database", "loopcheck.db")
if not Path(db_path).exists():
    st.warning(f"No database at {db_path}. Run `loopcheck run --target targets/slugify` first.")
    st.stop()
conn = connect(db_path)

runs_tab, calib_tab, audit_tab = st.tabs(["Runs", "Calibration", "Audit chain"])

with runs_tab:
    runs = conn.execute("SELECT * FROM runs ORDER BY started_ts DESC").fetchall()
    if not runs:
        st.info("No runs yet.")
    for run in runs:
        with st.expander(
            f"{run['run_id']} — {run['target']} — {run['status']} "
            f"({run['iterations']} iterations, ${run['cost_usd'] or 0:.4f})"
        ):
            records = show_run(conn, run["run_id"])
            if records:
                st.line_chart(
                    {"confidence": [r["confidence"] for r in records]}, height=200
                )
                for r in records:
                    st.markdown(
                        f"**iteration {r['iteration']}** — confidence "
                        f"`{r['confidence']}` → **{r['decision']}**"
                    )
                    st.json(r["checks"], expanded=False)
            spans = conn.execute(
                "SELECT name, (end_ns - start_ns) / 1e6 AS ms, attrs FROM spans "
                "WHERE run_id = ? OR trace_id IN "
                "(SELECT trace_id FROM spans WHERE run_id = ?) ORDER BY start_ns",
                (run["run_id"], run["run_id"]),
            ).fetchall()
            if spans:
                st.caption("Trace")
                st.dataframe(
                    [{"span": s["name"], "duration_ms": round(s["ms"], 1)} for s in spans],
                    use_container_width=True,
                )

with calib_tab:
    report_path = Path("calibration/report.json")
    if not report_path.exists():
        st.info("Run `loopcheck calibrate` to produce calibration/report.json.")
    else:
        rep = json.loads(report_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{rep['precision']:.2f}")
        c2.metric("Recall", f"{rep['recall']:.2f}")
        c3.metric("F1", f"{rep['f1']:.2f}")
        c4.metric("Labeled files", rep["n"])
        st.subheader("Reliability")
        valid = [b for b in rep["bins"] if b["count"]]
        st.bar_chart(
            {f"{b['lo']:.1f}–{b['hi']:.1f}": b["frac_good"] for b in valid}
        )
        st.caption("Fraction of files labeled good, per confidence bin. "
                   "A calibrated verifier trends upward.")
        st.subheader("Per-file results")
        st.dataframe(rep["rows"], use_container_width=True)

with audit_tab:
    ok, bad_seq = verify_chain(conn, Config().audit_key())
    n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    if ok:
        st.success(f"HMAC chain intact — {n} records verified")
    else:
        st.error(f"HMAC chain BROKEN at seq {bad_seq}")
    st.dataframe(
        [dict(r) for r in conn.execute(
            "SELECT seq, run_id, iteration, substr(hmac, 1, 16) AS hmac_prefix "
            "FROM audit_log ORDER BY seq"
        )],
        use_container_width=True,
    )
```

- [ ] **Step 2: Smoke test**

Run: `uv run streamlit run dashboard/app.py --server.headless true` — confirm it boots without traceback (Ctrl-C after "You can now view"). With no DB present it must show the warning, not crash.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit dashboard - runs, calibration, audit chain"
```

---

### Task 16: README, docker-compose (Jaeger), final verification

**Files:**
- Create: `README.md`, `docker-compose.yml`

- [ ] **Step 1: Write docker-compose.yml** (optional Jaeger for OTLP demo):

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.60
    ports:
      - "16686:16686"   # UI
      - "4318:4318"     # OTLP HTTP
```

- [ ] **Step 2: Write README.md** — must contain, in this order: the one-line thesis ("Most loop engineering demos measure whether the agent finished. loopcheck measures whether you should trust that it finished."); architecture diagram (ASCII) of generate → verify → decide with the four layers; quickstart (`uv sync`, `export ANTHROPIC_API_KEY=...`, `export LOOPCHECK_AUDIT_KEY=...`, `uv run loopcheck run --target targets/slugify`, `uv run loopcheck calibrate`, `uv run loopcheck dashboard`); a "How the verifier works" section (signals + weights table, thresholds, why coverage is weighted low — it's gameable); a "Measuring the verifier itself" section explaining the labeled set and precision/recall; an "Audit trail" section with the HMAC chain construction and the tamper demo (`sqlite3 loopcheck.db "UPDATE audit_log SET payload='{}' WHERE seq=1"` then `loopcheck audit verify` → BROKEN); "Honest limitations" section: generated code runs unsandboxed (subprocess+timeout only — do not point at untrusted targets), judge shares a vendor with the agent, seed calibration set is small (15 files) until expanded via scripts/gen_flawed.py, coverage/mutation signals only as good as the mutation operators. Optional Jaeger: `docker compose up -d` then `loopcheck run` with `Config(otlp_endpoint="http://localhost:4318/v1/traces")` via `--otlp` flag if added, else document code-level config.

- [ ] **Step 3: Full verification**

Run: `uv run pytest` → all PASS. `uv run ruff check src tests targets dashboard scripts` → clean. `uv run loopcheck --help`, `uv run loopcheck audit verify` → work.

- [ ] **Step 4: Commit**

```bash
git add README.md docker-compose.yml
git commit -m "docs: README with thesis, quickstart, honest limitations; optional Jaeger"
```

---

## Post-plan (requires ANTHROPIC_API_KEY, run interactively with the user)

Not part of the automated plan — these produce the demo artifacts:
1. `uv run loopcheck run --target targets/slugify` (repeat for pricing, ratelimit) — real runs for the dashboard.
2. `uv run loopcheck calibrate` — real report including the judge signal.
3. Expand calibration set toward ~50 files with `scripts/gen_flawed.py` + hand review.
4. Vendor one real-world module as a fourth target (showcase).
