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
    raw_criteria = data.get("criteria", {})
    criteria = {k: float(raw_criteria.get(k, 0.0)) for k in CRITERIA}
    return JudgeResult(
        score=sum(criteria.values()) / len(criteria),
        criteria=criteria,
        rationale=str(data.get("rationale", "")),
        cost_usd=resp.cost_usd,
    )
