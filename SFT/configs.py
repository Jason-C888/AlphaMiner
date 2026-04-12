from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_M1_INPUT_CANDIDATES = (
    "extracter/output/samples.jsonl",
    "extracter/output/samples2.jsonl",
    "extracter/output/samples_filtered.jsonl",
    "extracter/output/samples_stable.jsonl",
)
DEFAULT_M2_INPUT_CANDIDATES = (
    "SFT/data/prepared_samples.jsonl",
)
DEFAULT_DATA_DIR = "SFT/data"
DEFAULT_OUTPUT_DIR = "SFT/output"


@dataclass(frozen=True)
class RuntimeConfig:
    stage: str
    input_path: Path
    data_dict_path: Path
    data_dir: Path
    output_dir: Path
    version_id: str


def build_runtime_config(
    *,
    stage: str,
    input_path: str | Path | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    version_id: str | None = None,
) -> RuntimeConfig:
    project_root = Path(__file__).resolve().parent.parent
    resolved_input_path = _resolve_input_path(project_root, stage, input_path)
    resolved_data_dir = _resolve_path(project_root, data_dir)
    resolved_output_dir = _resolve_path(project_root, output_dir)
    return RuntimeConfig(
        stage=stage,
        input_path=resolved_input_path,
        data_dict_path=project_root / "extracter/data_dict.md",
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
        version_id=version_id or _default_version_id(),
    )


def _resolve_input_path(
    project_root: Path,
    stage: str,
    input_path: str | Path | None,
) -> Path:
    if input_path is not None:
        resolved = _resolve_path(project_root, input_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Input JSONL file not found: {resolved}")
        return resolved

    default_candidates = (
        DEFAULT_M1_INPUT_CANDIDATES if stage == "m1" else DEFAULT_M2_INPUT_CANDIDATES
    )
    for candidate in default_candidates:
        resolved = project_root / candidate
        if resolved.exists():
            return resolved
    searched = ", ".join(default_candidates)
    raise FileNotFoundError(f"No extracter sample file found. Searched: {searched}")


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return project_root / resolved


def _default_version_id() -> str:
    return f"sft_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
