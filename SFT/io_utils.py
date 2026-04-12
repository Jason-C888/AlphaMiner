from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> list[str]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")
    return resolved.read_text(encoding="utf-8").splitlines()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    resolved = Path(path)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    resolved = Path(path)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
