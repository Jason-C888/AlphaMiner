from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from extracter.parser.data_dict_parser import load_data_dictionary

from .configs import RuntimeConfig
from .data_builder import prepare_dataset
from .evaluator import evaluate_records
from .inference_backends import build_inference_backend
from .inference_config import InferenceConfig
from .io_utils import ensure_directory, read_jsonl, write_json, write_jsonl
from .prompt_builder import build_allowed_fields_text, build_inference_messages
from .training_data_builder import build_chat_splits


@dataclass(frozen=True)
class PipelineResult:
    stage: str
    version: str
    input_path: str
    summary_path: str
    output_paths: dict[str, str]
    record_count: int
    payload: dict | None = None


def run_m1_pipeline(config: RuntimeConfig) -> PipelineResult:
    ensure_directory(config.data_dir)
    ensure_directory(config.output_dir)

    lines = read_jsonl(config.input_path)
    dataset = prepare_dataset(
        lines,
        version_id=config.version_id,
        source_name=config.input_path.name,
    )

    prepared_samples_path = config.data_dir / "prepared_samples.jsonl"
    review_records_path = config.output_dir / "review_records.jsonl"
    summary_path = config.output_dir / "m1_summary.json"

    write_jsonl(prepared_samples_path, dataset.prepared_samples)
    write_jsonl(review_records_path, dataset.review_records)
    write_json(
        summary_path,
        {
            **dataset.summary,
            "input_path": str(config.input_path),
            "prepared_samples_path": str(prepared_samples_path),
            "review_records_path": str(review_records_path),
            "summary_path": str(summary_path),
        },
    )

    return PipelineResult(
        stage="m1",
        version=config.version_id,
        input_path=str(config.input_path),
        summary_path=str(summary_path),
        output_paths={
            "prepared_samples_path": str(prepared_samples_path),
            "review_records_path": str(review_records_path),
        },
        record_count=len(dataset.prepared_samples),
    )


def run_m2_pipeline(config: RuntimeConfig) -> PipelineResult:
    ensure_directory(config.data_dir)
    ensure_directory(config.output_dir)

    lines = read_jsonl(config.input_path)
    dataset = build_chat_splits(
        lines,
        data_dict_path=str(config.data_dict_path),
        source_name=config.input_path.name,
    )

    train_path = config.data_dir / "train.jsonl"
    val_path = config.data_dir / "val.jsonl"
    test_path = config.data_dir / "test.jsonl"
    split_manifest_path = config.output_dir / "split_manifest.jsonl"
    summary_path = config.output_dir / "m2_summary.json"

    write_jsonl(train_path, dataset.split_records["train"])
    write_jsonl(val_path, dataset.split_records["val"])
    write_jsonl(test_path, dataset.split_records["test"])
    write_jsonl(split_manifest_path, dataset.split_manifest)
    write_json(
        summary_path,
        {
            **dataset.summary,
            "input_path": str(config.input_path),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "split_manifest_path": str(split_manifest_path),
            "summary_path": str(summary_path),
        },
    )

    return PipelineResult(
        stage="m2",
        version=dataset.summary["versions"][0] if dataset.summary["versions"] else config.version_id,
        input_path=str(config.input_path),
        summary_path=str(summary_path),
        output_paths={
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "split_manifest_path": str(split_manifest_path),
        },
        record_count=sum(len(records) for records in dataset.split_records.values()),
    )


def run_infer_pipeline(
    *,
    inference_config: InferenceConfig,
    inspiration: str,
    output_dir: str | Path,
    save_path: str | Path | None = None,
) -> PipelineResult:
    resolved_output_dir = ensure_directory(output_dir)
    data_dictionary = load_data_dictionary(inference_config.data_dict_path)
    allowed_fields_text = build_allowed_fields_text(data_dictionary)
    messages = build_inference_messages(
        inspiration=inspiration,
        allowed_fields_text=allowed_fields_text,
    )

    backend = build_inference_backend(inference_config)
    response = backend.generate(messages)
    result_payload = response.parsed_json or {
        "backend_type": response.backend_type,
        "model_name": response.model_name,
        "input_inspiration": inspiration,
        "parse_error": response.parse_error,
        "raw_text": response.raw_text,
    }

    output_paths: dict[str, str] = {}
    summary_path = ""
    if save_path is not None:
        resolved_save_path = Path(save_path)
        if not resolved_save_path.is_absolute():
            resolved_save_path = (resolved_output_dir / resolved_save_path).resolve()
        ensure_directory(resolved_save_path.parent)
        write_json(resolved_save_path, result_payload)
        output_paths["infer_result_path"] = str(resolved_save_path)
        summary_path = str(resolved_save_path)

    return PipelineResult(
        stage="infer",
        version="inference",
        input_path="<inline inspiration>",
        summary_path=summary_path,
        output_paths=output_paths,
        record_count=1,
        payload=result_payload,
    )


def run_eval_pipeline(
    *,
    inference_config: InferenceConfig,
    input_path: str | Path,
    output_dir: str | Path,
) -> PipelineResult:
    resolved_output_dir = ensure_directory(output_dir)
    resolved_input_path = Path(input_path).resolve()
    lines = read_jsonl(resolved_input_path)
    backend = build_inference_backend(inference_config)
    result = evaluate_records(
        lines,
        backend=backend,
        inference_config=inference_config,
    )

    eval_records_path = resolved_output_dir / "eval_records.jsonl"
    eval_report_path = resolved_output_dir / "eval_report.json"
    write_jsonl(eval_records_path, result.records)
    write_json(
        eval_report_path,
        {
            **result.summary,
            "input_path": str(resolved_input_path),
            "eval_records_path": str(eval_records_path),
            "eval_report_path": str(eval_report_path),
        },
    )

    return PipelineResult(
        stage="eval",
        version="inference",
        input_path=str(resolved_input_path),
        summary_path=str(eval_report_path),
        output_paths={
            "eval_records_path": str(eval_records_path),
            "eval_report_path": str(eval_report_path),
        },
        record_count=len(result.records),
    )


def asdict_result(result: PipelineResult) -> dict[str, str | int]:
    return asdict(result)
