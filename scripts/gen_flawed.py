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
