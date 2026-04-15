from __future__ import annotations

from pathlib import Path


_MODEL_CONFIG_FILES = ("config.json", "configuration.json")
_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def has_local_model_files(local_model_dir: str | Path) -> bool:
    resolved_dir = Path(local_model_dir).resolve()
    if not resolved_dir.is_dir():
        return False
    has_config = any((resolved_dir / filename).exists() for filename in _MODEL_CONFIG_FILES)
    has_weights = any(any(resolved_dir.glob(pattern)) for pattern in _WEIGHT_PATTERNS)
    return has_config and has_weights


def download_model_to_local(
    *,
    model_id: str,
    local_model_dir: str | Path,
    revision: str | None = None,
) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope is required for model downloads. Install it with `pip install modelscope`."
        ) from exc

    resolved_dir = Path(local_model_dir).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id=model_id,
        local_dir=str(resolved_dir),
        revision=revision,
    )
    return resolved_dir


def ensure_local_model(
    *,
    model_id: str,
    local_model_dir: str | Path,
    revision: str | None = None,
) -> Path:
    resolved_dir = Path(local_model_dir).resolve()
    if has_local_model_files(resolved_dir):
        return resolved_dir
    download_model_to_local(
        model_id=model_id,
        local_model_dir=resolved_dir,
        revision=revision,
    )
    return resolved_dir
