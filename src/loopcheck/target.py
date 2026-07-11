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
