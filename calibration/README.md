# Calibration set

Hand-labeled test files used to measure the verifier itself (precision/recall of
"accept"). Naming: `<target>/<label>__<slug>.py`, label ∈ {good, bad}.

- good: genuinely verifies the target's SPEC.md behavior
- bad: passes or superficially plausible but does NOT verify intended behavior
  (assertion-free, tautological, wrong-behavior, target-mocked-away)

Seeds are hand-written. Expand with `uv run python scripts/gen_flawed.py <target>`
(uses the Anthropic API), then hand-review every generated file before trusting
the labels — the label is the ground truth being measured against.
