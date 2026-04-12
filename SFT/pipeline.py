from __future__ import annotations

from dataclasses import asdict, dataclass

from .configs import RuntimeConfig
from .data_builder import prepare_dataset
from .io_utils import ensure_directory, read_jsonl, write_json, write_jsonl
from .training_data_builder import build_chat_splits


@dataclass(frozen=True)
class PipelineResult:
    stage: str
    version: str
    input_path: str
    summary_path: str
    output_paths: dict[str, str]
    record_count: int


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


def asdict_result(result: PipelineResult) -> dict[str, str | int]:
    return asdict(result)
