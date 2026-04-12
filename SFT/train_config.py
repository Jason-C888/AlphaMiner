from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    base_model_name_or_path: str
    trust_remote_code: bool = True
    use_lora: bool = True
    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    target_modules: list[str] = field(default_factory=list)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass(frozen=True)
class DataConfig:
    train_file: Path
    val_file: Path
    test_file: Path | None
    max_prompt_length: int
    max_seq_length: int
    completion_only_loss: bool = True
    assistant_only_loss: bool = False
    packing: bool = False
    preprocessing_num_workers: int = 1


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    logging_dir: Path
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    run: RunConfig
    config_path: Path


def load_train_config(path: str | Path) -> TrainConfig:
    resolved_path = Path(path).resolve()
    raw = _load_yaml(resolved_path)

    model = ModelConfig(**raw["model"])
    data_section = raw["data"]
    run_section = raw["run"]
    config_dir = resolved_path.parent
    data = DataConfig(
        train_file=_resolve_path(config_dir, data_section["train_file"]),
        val_file=_resolve_path(config_dir, data_section["val_file"]),
        test_file=_resolve_path(config_dir, data_section["test_file"])
        if data_section.get("test_file")
        else None,
        max_prompt_length=int(data_section["max_prompt_length"]),
        max_seq_length=int(data_section["max_seq_length"]),
        completion_only_loss=bool(data_section.get("completion_only_loss", True)),
        assistant_only_loss=bool(data_section.get("assistant_only_loss", False)),
        packing=bool(data_section.get("packing", False)),
        preprocessing_num_workers=int(data_section.get("preprocessing_num_workers", 1)),
    )
    run = RunConfig(
        output_dir=_resolve_path(config_dir, run_section["output_dir"]),
        logging_dir=_resolve_path(config_dir, run_section["logging_dir"]),
        num_train_epochs=float(run_section["num_train_epochs"]),
        per_device_train_batch_size=int(run_section["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(run_section["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(run_section["gradient_accumulation_steps"]),
        learning_rate=float(run_section["learning_rate"]),
        warmup_ratio=float(run_section["warmup_ratio"]),
        weight_decay=float(run_section["weight_decay"]),
        logging_steps=int(run_section["logging_steps"]),
        eval_steps=int(run_section["eval_steps"]),
        save_steps=int(run_section["save_steps"]),
        save_total_limit=int(run_section["save_total_limit"]),
        seed=int(run_section["seed"]),
        bf16=bool(run_section.get("bf16", True)),
        fp16=bool(run_section.get("fp16", False)),
        gradient_checkpointing=bool(run_section.get("gradient_checkpointing", True)),
        lr_scheduler_type=str(run_section.get("lr_scheduler_type", "cosine")),
        report_to=list(run_section.get("report_to", [])),
    )
    return TrainConfig(
        model=model,
        data=data,
        run=run,
        config_path=resolved_path,
    )


def _resolve_path(base_dir: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (base_dir / resolved).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the training config. Install pyyaml in the training environment."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid training config format: {path}")
    return payload
