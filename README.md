# loopcheck

Most loop engineering demos measure whether the agent finished. loopcheck measures whether you should trust that it finished.

---

## What it is

loopcheck is a verifier-first agent loop for test generation. The agent writes pytest tests for a target module; a multi-signal verifier scores them; the loop decides to accept, retry with feedback, or escalate. Every verification is recorded in a tamper-evident audit chain. The verifier is itself calibrated on a labeled seed set so you can measure its precision and recall.

The four layers, in order of implementation:

1. **Verifier** — five objective signals plus an optional LLM judge produce a scalar confidence score.
2. **Loop** — a LangGraph state machine drives generate → verify → decide, with SQLite checkpointing for resume.
3. **Observability** — OpenTelemetry spans written to SQLite; optional OTLP export to Jaeger or any collector.
4. **Audit chain** — HMAC-SHA256 linked list over every verification record; detects modification or reordering of existing records.

---

## Architecture

```
                    ┌─────────────┐
  target/           │   generate  │  <── Claude (agent_model)
  module.py  ──────>│   (LangGraph│
  SPEC.md           │    node)    │
                    └──────┬──────┘
                           │ test_code
                           v
                    ┌─────────────┐
                    │   verify    │
                    │   (LangGraph│
                    │    node)    │
                    └──────┬──────┘
                           │ Verdict(confidence, decision, feedback)
                           │
              ┌────────────┼────────────┐
              v            v            v
           accept        retry       escalate
                      (feedback ──>
                        generate)

  Verifier internals:
  ┌──────────────────────────────────────────────────────┐
  │  check_mutation      kill_rate          weight 0.45  │
  │  check_tests_pass    0 or 1.0           weight 0.20  │
  │  judge_tests         LLM score 0..1    weight 0.15  │
  │  check_assertions    fraction w/ assert weight 0.10  │
  │  check_no_target_mock 0 or 1.0         weight 0.05  │
  │  check_coverage      branch pct        weight 0.05  │
  │                                                      │
  │  confidence = weighted average of present signals    │
  │  accept >= 0.85  /  retry  /  escalate < 0.50        │
  │       or stagnation (two non-improving iterations)   │
  └──────────────────────────────────────────────────────┘

  Every Verdict is appended to the HMAC-SHA256 audit chain:
  genesis(000...0) → HMAC(key, prev||payload) → HMAC(...) → ...

  OTel spans: generate, verify, check:* written to SQLite spans table.
  Optional: BatchSpanProcessor → OTLP HTTP → Jaeger.
```

---

## Quickstart

**Requirements:** Python >=3.12, [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd loopcheck

uv sync

export ANTHROPIC_API_KEY=sk-ant-...
export LOOPCHECK_AUDIT_KEY=$(openssl rand -hex 32)

# Run the loop on a target
uv run loopcheck run --target targets/slugify

# Resume an interrupted run
uv run loopcheck run --target targets/slugify --resume <run_id>

# Cap iterations (useful for dry-runs)
uv run loopcheck run --target targets/slugify --max-iterations 2

# Evaluate the verifier on the labeled calibration set
uv run loopcheck calibrate

# Same but skip the LLM judge (no API key required)
uv run loopcheck calibrate --no-judge

# Verify the audit chain has not been tampered with
uv run loopcheck audit verify

# Inspect a specific run's records
uv run loopcheck audit show <run_id>

# Launch the Streamlit dashboard
uv run loopcheck dashboard
```

**Output of a successful run:**

```
run_id      a3f9c12e8b04
decision    accept
iterations  2
confidence  0.891
cost_usd    0.0143
```

---

## How the verifier works

The verifier computes a weighted confidence score from five objective signals and one optional LLM judge.

### Signals and weights

| Signal | Weight | What it measures |
|---|---|---|
| `mutation` | 0.45 | Kill rate: fraction of auto-generated mutants the tests detect |
| `tests_pass` | 0.20 | 1.0 if all tests pass, 0.0 if any fail or error |
| `judge` | 0.15 | LLM score (0–1) on spec coverage, edge cases, assertion quality |
| `assertions` | 0.10 | Fraction of test functions containing an `assert` or `pytest.raises` |
| `no_target_mock` | 0.05 | 1.0 if the module under test is not mocked; 0.0 otherwise |
| `coverage` | 0.05 | Branch coverage percentage |

Weights are configured in `Config.weights` and can be overridden per run.

### Thresholds and decisions

- **accept** — confidence >= 0.85
- **escalate** — confidence < 0.50, or two consecutive non-improving iterations (stagnation)
- **retry** — everything else; verifier feedback is passed back to the agent

### Why coverage is weighted so low

Branch coverage is gameable: `assert True` achieves 100% coverage while testing nothing. Coverage is included as a weak secondary signal only. The mutation kill rate is the primary proxy for test effectiveness because it measures whether the tests actually detect behavioral changes in the code.

### Mutation operators

Mutations applied to the target source (up to `max_mutants=20` per run):

- Flip comparison operators (`<` ↔ `<=`, `>` ↔ `>=`, `==` ↔ `!=`)
- Off-by-one on integer constants (`n` → `n+1`)
- Negate `if`-conditions (wrap test in `not`)
- Swap operands of `-`, `/`, `//`, `%`
- Replace `if`-bodies with `pass` (delete branch)
- Flip boolean literals (`True` ↔ `False`)

A mutant that all tests pass against is a "survivor" — evidence the tests are not catching a class of bugs.

---

## Measuring the verifier itself

The calibration subsystem answers: how reliable is the confidence score?

### The labeled set

`calibration/<target>/` contains hand-labeled pytest files named `good__*.py` or `bad__*.py`. The seed set has 15 files across the three targets (slugify, pricing, ratelimit). Expand it with:

```bash
python scripts/gen_flawed.py   # generates additional bad__* variants
```

### Running calibration

```bash
uv run loopcheck calibrate          # uses the LLM judge
uv run loopcheck calibrate --no-judge   # objective signals only
```

Output includes precision, recall, F1, and a per-file table showing predicted vs. labeled.

### Measured results (no-judge run)

On the 15-file seed set without the LLM judge:

- Mean confidence of **good**-labeled files: **0.837**
- Mean confidence of **bad**-labeled files: **0.418**
- Precision at the 0.85 accept threshold: **1.00**
- Recall at the 0.85 accept threshold: **0.33**

Note: `coverage` and `mutation` are scored 0 and skipped when tests fail against the real (unmutated) module, so a test file with failing assertions cannot get an artificially inflated mutation score.

The ranking is correct — good files score higher than bad on average — but recall is poor because several genuinely good test files score below 0.85 on objective signals alone. This is the calibration layer doing its job: it reveals that objective signals alone are insufficient at the accept threshold and that the judge signal is needed to recover recall. Full judge-inclusive numbers require an API key and are pending real runs.

---

## Audit trail

Every time the verifier runs, it appends a JSON record to an HMAC-SHA256 linked list stored in `audit_log` in the SQLite database.

### Chain construction

```
record_1: { run_id, iteration, ts, confidence, decision, checks, rationale_sha256 }
hmac_1   = HMAC-SHA256(key, genesis_hash || JSON(record_1))

record_2: { ... }
hmac_2   = HMAC-SHA256(key, hmac_1 || JSON(record_2))
```

The genesis hash is `"0" * 64`. Each record's HMAC covers its own payload and the previous HMAC, so any modification to any record or any reordering breaks all subsequent HMACs.

The key is read from `LOOPCHECK_AUDIT_KEY` (environment variable). Without it, a dev fallback is used with a logged warning. Set a real key before storing records you care about.

### Tamper detection demo

```bash
# Verify a clean chain
uv run loopcheck audit verify
# chain OK (47 records)

# Tamper with a record
sqlite3 loopcheck.db "UPDATE audit_log SET payload='{}' WHERE seq=1"

# Re-verify
uv run loopcheck audit verify
# chain BROKEN at seq 1
```

---

## Optional: Jaeger traces

Jaeger provides a UI for browsing the OTel spans from each run.

```bash
docker compose up -d
# UI at http://localhost:16686
# OTLP HTTP at http://localhost:4318
```

To export spans to Jaeger, set `otlp_endpoint` in `Config`:

```python
from loopcheck.config import Config
config = Config(otlp_endpoint="http://localhost:4318/v1/traces")
```

There is no CLI flag for `--otlp`; wire it via code or a thin wrapper. Without `otlp_endpoint`, spans are written only to the local SQLite `spans` table, which is sufficient for the dashboard and audit.

---

## Targets

Three reference targets are included, each with a `module.py` and `SPEC.md`:

| Target | What it does |
|---|---|
| `targets/slugify` | Convert arbitrary strings to URL-safe slugs |
| `targets/pricing` | Tier-based pricing calculator with volume discounts |
| `targets/ratelimit` | Token-bucket rate limiter with configurable capacity |

Point `--target` at any directory containing `module.py` and `SPEC.md`.

---

## Honest limitations

- **Generated code runs unsandboxed.** Tests are executed via `subprocess` with a timeout. Do not point loopcheck at untrusted targets.
- **The judge shares a vendor with the agent.** Both use Anthropic models. The judge is not an independent evaluator; it may have correlated blind spots.
- **The seed calibration set is small.** 15 labeled files is enough to observe signal behavior but not enough to estimate precision/recall reliably. Expand with `scripts/gen_flawed.py` and hand review before drawing conclusions.
- **Mutation operators bound what "tested" means.** The kill rate measures only the specific mutation classes the tool generates. Tests can score 1.0 on mutation while missing entire behavioral dimensions the operators do not cover.
- **The audit chain cannot detect truncation.** The HMAC chain detects modification or reordering of existing records, but it cannot detect deletion of trailing records or erasure of the entire table, because the head of the chain is not externally anchored. Do not rely on it as a sole integrity control.
- **Python >=3.12 required.** The development environment resolved Python 3.13. The package declares `requires-python = ">=3.12"` and is not tested on older versions.

---

## Development

```bash
uv run pytest                              # ~70 tests, ~90 seconds
uv run ruff check src tests targets dashboard scripts
```

Tests cover each signal independently, the verifier pipeline, the audit chain, the LangGraph loop (with mocked LLM), calibration metrics, and CLI entry points.
