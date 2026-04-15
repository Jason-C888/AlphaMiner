from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_manager import ensure_local_model
from .train_config import TrainConfig, load_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument(
        "--config",
        default="SFT/configs/train_config.yaml",
        help="Path to the YAML training config.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_train_config(args.config)
    train(config)
    return 0


def train(config: TrainConfig) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    config.run.output_dir.mkdir(parents=True, exist_ok=True)
    config.run.logging_dir.mkdir(parents=True, exist_ok=True)
    base_model_path = resolve_training_base_model_path(config)
    write_run_manifest(config, base_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_init_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.use_qlora and config.model.load_in_4bit:
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(config.model.bnb_4bit_compute_dtype),
        )
        model_init_kwargs["device_map"] = "auto"

    peft_config = None
    if config.model.use_lora:
        peft_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.model.target_modules,
        )

    training_args = SFTConfig(
        output_dir=str(config.run.output_dir),
        logging_dir=str(config.run.logging_dir),
        num_train_epochs=config.run.num_train_epochs,
        per_device_train_batch_size=config.run.per_device_train_batch_size,
        per_device_eval_batch_size=config.run.per_device_eval_batch_size,
        gradient_accumulation_steps=config.run.gradient_accumulation_steps,
        learning_rate=config.run.learning_rate,
        warmup_ratio=config.run.warmup_ratio,
        weight_decay=config.run.weight_decay,
        logging_steps=config.run.logging_steps,
        eval_steps=config.run.eval_steps,
        save_steps=config.run.save_steps,
        save_total_limit=config.run.save_total_limit,
        bf16=config.run.bf16,
        fp16=config.run.fp16,
        gradient_checkpointing=config.run.gradient_checkpointing,
        lr_scheduler_type=config.run.lr_scheduler_type,
        report_to=config.run.report_to,
        seed=config.run.seed,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        dataset_num_proc=config.data.preprocessing_num_workers,
        max_length=config.data.max_seq_length,
        completion_only_loss=config.data.completion_only_loss,
        assistant_only_loss=config.data.assistant_only_loss,
        packing=config.data.packing,
        model_init_kwargs=model_init_kwargs,
    )

    train_dataset = load_dataset(
        "json",
        data_files=str(config.data.train_file),
        split="train",
    )
    eval_dataset = load_dataset(
        "json",
        data_files=str(config.data.val_file),
        split="train",
    )

    trainer = SFTTrainer(
        model=str(base_model_path),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.run.output_dir)


def resolve_training_base_model_path(config: TrainConfig) -> Path:
    return ensure_local_model(
        model_id=config.model.model_id,
        local_model_dir=config.model.local_model_dir,
        revision=config.model.revision,
    )


def write_run_manifest(config: TrainConfig, base_model_path: Path) -> None:
    manifest_path = config.run.output_dir / "run_manifest.json"
    payload = {
        "config_path": str(config.config_path),
        "model_id": config.model.model_id,
        "local_model_dir": str(config.model.local_model_dir),
        "resolved_base_model_path": str(base_model_path),
        "train_file": str(config.data.train_file),
        "val_file": str(config.data.val_file),
        "test_file": str(config.data.test_file) if config.data.test_file else None,
        "output_dir": str(config.run.output_dir),
        "logging_dir": str(config.run.logging_dir),
        "max_prompt_length": config.data.max_prompt_length,
        "max_seq_length": config.data.max_seq_length,
        "completion_only_loss": config.data.completion_only_loss,
        "assistant_only_loss": config.data.assistant_only_loss,
        "packing": config.data.packing,
        "trainer_framework": "trl",
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_torch_dtype(name: str):
    import torch

    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


if __name__ == "__main__":
    raise SystemExit(main())
