from __future__ import annotations

import argparse
import json

from .configs import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, build_runtime_config
from .download_model import download_from_config
from .inference_config import DEFAULT_INFERENCE_CONFIG_PATH, load_inference_config
from .pipeline import (
    asdict_result,
    run_eval_pipeline,
    run_infer_pipeline,
    run_m1_pipeline,
    run_m2_pipeline,
)
from .trainer import train
from .train_config import load_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT pipeline entrypoint.")
    parser.add_argument(
        "--stage",
        choices=("m1", "m2", "m3", "download", "infer", "eval"),
        required=True,
        help="SFT stage to run.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to the input JSONL for m1/m2/eval. Defaults to stage-specific known candidates.",
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
    parser.add_argument(
        "--inference-config",
        default=DEFAULT_INFERENCE_CONFIG_PATH,
        help="YAML config path used by stages infer/eval.",
    )
    parser.add_argument(
        "--inspiration",
        default=None,
        help="Phenomenon description used by stage infer.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional JSON output path used by stage infer.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage in {"m3", "download"}:
        train_config = load_train_config(args.train_config)
        if args.stage == "download":
            local_model_dir = download_from_config(train_config)
            print(
                json.dumps(
                    {
                        "stage": "download",
                        "config_path": str(train_config.config_path),
                        "model_id": train_config.model.model_id,
                        "local_model_dir": str(local_model_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        train(train_config)
        print(
            json.dumps(
                {
                    "stage": "m3",
                    "config_path": str(train_config.config_path),
                    "model_id": train_config.model.model_id,
                    "local_model_dir": str(train_config.model.local_model_dir),
                    "output_dir": str(train_config.run.output_dir),
                    "trainer_framework": "trl",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.stage in {"infer", "eval"}:
        inference_config = load_inference_config(args.inference_config)
        if args.stage == "infer":
            if not args.inspiration:
                parser.error("--inspiration is required when --stage infer")
            result = run_infer_pipeline(
                inference_config=inference_config,
                inspiration=args.inspiration,
                output_dir=args.output_dir,
                save_path=args.save_path,
            )
            print(json.dumps(result.payload, ensure_ascii=False, indent=2))
            return 0
        else:
            input_path = args.input_path or inference_config.default_eval_input_path
            result = run_eval_pipeline(
                inference_config=inference_config,
                input_path=input_path,
                output_dir=args.output_dir,
            )
        print(json.dumps(asdict_result(result), ensure_ascii=False, indent=2))
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
