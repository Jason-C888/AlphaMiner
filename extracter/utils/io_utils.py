from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FailureRecord:
    stage: str
    report_title: str
    reason_type: str
    reason: str


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_candidates_csv(path: str | Path, rows: Iterable[object]) -> None:
    resolved = Path(path)
    fieldnames = [
        "report_title",
        "score",
        "rank",
        "report_path",
        "report_date",
        "broker",
        "text_length",
        "keyword_signal_count",
        "section_signal_count",
        "candidate_section_count",
        "garble_ratio",
    ]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_failures_csv(path: str | Path, rows: Iterable[FailureRecord]) -> None:
    resolved = Path(path)
    fieldnames = ["stage", "report_title", "reason_type", "reason"]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_candidates_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    resolved = Path(path)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
