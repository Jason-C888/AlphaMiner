from __future__ import annotations

import argparse
import json

from .configs import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, build_runtime_config
from .pipeline import asdict_result, run_m1_pipeline, run_m2_pipeline
from .trainer import train
from .train_config import load_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT pipeline entrypoint.")
    parser.add_argument(
        "--stage",
        choices=("m1", "m2", "m3"),
        required=True,
        help="SFT stage to run.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to extracter sample JSONL. Defaults to the first available known candidate.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory for SFT data outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SFT summary outputs.",
    )
    parser.add_argument(
        "--version-id",
        default=None,
        help="Optional explicit version id. Defaults to sft_<timestamp>.",
    )
    parser.add_argument(
        "--train-config",
        default="SFT/configs/train_config.yaml",
        help="YAML config path used by stage m3.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "m3":
        train_config = load_train_config(args.train_config)
        train(train_config)
        print(
            json.dumps(
                {
                    "stage": "m3",
                    "config_path": str(train_config.config_path),
                    "output_dir": str(train_config.run.output_dir),
                    "trainer_framework": "trl",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = build_runtime_config(
        stage=args.stage,
        input_path=args.input_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        version_id=args.version_id,
    )
    if args.stage == "m1":
        result = run_m1_pipeline(config)
    else:
        result = run_m2_pipeline(config)
    print(json.dumps(asdict_result(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
