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
