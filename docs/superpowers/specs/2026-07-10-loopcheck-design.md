# loopcheck — Design Spec

**Date:** 2026-07-10
**Status:** Approved by user (brainstorming session)

## Thesis

Most agent-loop demos measure whether the agent finished. loopcheck measures whether you should trust that it finished — and then evaluates the trust mechanism itself by reporting verifier precision/recall against hand-labeled data. The "trust" claim is measured, not asserted.

**Demo task:** a self-verifying test-writing loop. An agent (Claude) writes pytest tests for a target Python module; a confidence-scored verifier decides whether the tests actually test the intended behavior — not merely whether they pass.

## Architecture — four layers

### 1. Loop runtime (LangGraph)

- A `StateGraph` with three nodes:
  - **generate** — Claude writes (or revises, given verifier feedback) a pytest test file for the target module.
  - **verify** — the verifier layer scores the test file and produces a confidence score plus structured feedback.
  - **decide** — conditional edge routing: accept / retry-with-feedback / escalate-to-human / stop at iteration cap.
- **State:** target module path + behavior spec, current test code, verification history (all scores and feedback per iteration), iteration count, run id.
- **Persistence:** `langgraph-checkpoint-sqlite` (`SqliteSaver`). `loopcheck run --resume <run-id>` resumes a crashed or interrupted run from the last checkpoint.
- **Stopping conditions (explicit):**
  - confidence ≥ accept threshold → accept, stop
  - iteration count ≥ cap (default 5) → stop, report unconverged
  - confidence < escalation threshold, or scores oscillating (no improvement over 2 consecutive iterations) → escalate: flag for human review, stop
- All decision logic lives in plain, unit-tested Python functions; graph nodes are thin wrappers.

### 2. Verifier layer (confidence-scored, not binary)

Combines objective and judged signals into a confidence score in [0, 1].

**Objective signals:**

- **Mutation testing (primary).** A hand-rolled AST mutation engine (~6 operators: flip comparison operator, off-by-one on integer constants, negate boolean condition, swap binary operands, delete conditional branch, replace constant). It generates mutants of the target module; kill rate = fraction of mutants that cause at least one test failure. Hand-rolled (rather than mutmut) for programmatic control and as an engineering signal.
- **Sanity checks:** tests collect and pass against the unmutated module; tests contain real assertions (AST scan); tests do not mock/patch the target module itself; branch coverage via coverage.py (weighted low — coverage is gameable, and the README says so).

**Judged signal:**

- **LLM judge (Claude).** Rubric-scored: does the test file assert the module's *intended* behavior per its behavior spec, including edge cases? Returns per-criterion scores and rationale; rationale stored verbatim in the audit log.

**Combination and thresholds:**

- Confidence = weighted combination of signal scores. Default weights favor mutation kill rate.
- accept ≥ 0.85; retry in [0.50, 0.85) with feedback synthesized from failed checks (surviving mutants, judge criticisms) fed to the next generate; escalate < 0.50 or oscillation.
- Weights and thresholds are config, tuned during calibration.

**Calibration (the headline artifact):**

- ~50 hand-labeled test files: genuinely good ones plus deliberately flawed ones (assertion-free, tautological, target-mocked-away, asserting wrong behavior), produced adversarially plus collected from real runs.
- Tune weights/thresholds on a tuning split; report precision, recall, F1, and a reliability diagram on a holdout split.
- `loopcheck calibrate` runs the verifier over the labeled set and emits the report.

### 3. Observability (OpenTelemetry)

- One trace per run; one span per iteration; child spans for generate, each verifier check, judge, decide.
- Span attributes: model, tokens in/out, cost USD, confidence, per-check scores, decision, rejection reasons.
- Exporters: SQLite span store always (dashboard requires zero infra); OTLP optional (docker-compose Jaeger for demo screenshots).

### 4. Audit trail (HMAC chain)

- Every verification decision appended to an `audit_log` SQLite table: payload (run id, iteration, per-check scores, confidence, decision, judge-rationale hash, timestamp) plus `hmac = HMAC(key, prev_hmac || payload)`.
- `loopcheck audit verify` walks the chain and pinpoints the first tampered record, if any.
- `loopcheck audit show <run-id>` replays every decision and why it was made.

## Demo targets

Each target ships with a behavior spec (a `SPEC.md` alongside the module) describing intended behavior and edge cases. The generate prompt and the LLM judge both read this spec; it is the ground truth for "intended behavior."

- Three purpose-built modules with genuine edge cases (reproducible for anyone cloning the repo):
  1. unicode-aware slugifier (normalization, empty/whitespace, length truncation)
  2. tiered pricing calculator (boundary tiers, discounts, rounding)
  3. token-bucket rate limiter (refill timing, burst, clock injection)
- One vendored real-world module as the showcase: "then I pointed it at real code."

## Dashboard (Streamlit)

- **Run browser:** iteration timeline, confidence trajectory, cost per iteration, what was rejected and why.
- **Calibration page:** precision/recall/F1, reliability diagram.
- **Audit page:** chain verification status, decision replay.

## CLI

```
loopcheck run --target <module> [--resume <run-id>]
loopcheck calibrate
loopcheck audit verify | show <run-id>
loopcheck dashboard
```

## Stack

Python 3.12, uv, LangGraph + langgraph-checkpoint-sqlite, anthropic SDK, pytest + coverage.py, hand-rolled AST mutation engine, OpenTelemetry SDK, SQLite, Streamlit, ruff. Models configurable: agent defaults to Sonnet, judge to Haiku or Sonnet (cost control).

## Error handling

- LLM API failures: retry with backoff; on exhaustion, checkpoint and exit resumable.
- Generated test code runs in a subprocess with a timeout. Not fully sandboxed — documented as a known limitation (targets are our own code; do not point at untrusted code).

## Testing loopcheck itself (TDD throughout)

- Mutation engine: unit tests per operator on fixture modules.
- Verifier scoring: unit tests with fixture test files of known quality.
- Audit chain: tamper-detection tests (modify a record, chain verification must fail at that record).
- Loop: integration test with a fake LLM (canned generate/judge responses) — zero API cost in CI.

## Honest metrics reported

Goal success rate over N runs, iterations to convergence, cost per iteration, verifier precision/recall on the holdout set — including failure cases.

## Build order

1. Target modules + mutation engine + automated checks
2. LLM judge + confidence combiner
3. LangGraph loop + checkpointing
4. OTel instrumentation + HMAC audit chain
5. Calibration set + metrics report
6. Streamlit dashboard + README/demo polish

## Out of scope (v1)

- Full sandboxing of generated code
- Documentation-verification loop (possible v2)
- Multi-agent generation, non-Python targets, hosted deployment
