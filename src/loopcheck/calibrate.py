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
