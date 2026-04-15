from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_manager import download_model_to_local
from .train_config import TrainConfig, load_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a base model to the configured local directory.")
    parser.add_argument(
        "--config",
        default="SFT/configs/train_config.yaml",
        help="Path to the YAML training config.",
    )
    return parser


def download_from_config(config: TrainConfig) -> Path:
    return download_model_to_local(
        model_id=config.model.model_id,
        local_model_dir=config.model.local_model_dir,
        revision=config.model.revision,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_train_config(args.config)
    resolved_dir = download_from_config(config)
    print(
        json.dumps(
            {
                "model_id": config.model.model_id,
                "local_model_dir": str(resolved_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
