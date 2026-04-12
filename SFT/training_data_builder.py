from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from extracter.parser.data_dict_parser import load_data_dictionary

from .prompt_builder import build_allowed_fields_text, build_prompt_completion_record


@dataclass(frozen=True)
class ChatSplitDataset:
    split_records: dict[str, list[dict[str, Any]]]
    split_manifest: list[dict[str, Any]]
    summary: dict[str, Any]


def build_chat_splits(lines: list[str], *, data_dict_path: str, source_name: str) -> ChatSplitDataset:
    prepared_samples = [json.loads(line) for line in lines if line.strip()]
    data_dictionary = load_data_dictionary(data_dict_path)
    allowed_fields_text = build_allowed_fields_text(data_dictionary)
    chat_entries = [
        {
            "sample": sample,
            "chat_record": build_prompt_completion_record(sample, allowed_fields_text),
        }
        for sample in prepared_samples
    ]

    split_assignments = assign_report_splits(prepared_samples)
    split_records = {"train": [], "val": [], "test": []}
    split_manifest: list[dict[str, Any]] = []

    for entry in chat_entries:
        sample = entry["sample"]
        split = split_assignments[sample["report_title"]]
        split_records[split].append(entry["chat_record"])
        split_manifest.append(
            {
                "sample_id": sample["sample_id"],
                "report_title": sample["report_title"],
                "report_date": sample.get("report_date"),
                "broker": sample.get("broker"),
                "class": sample.get("class"),
                "version": sample.get("version"),
                "split": split,
            }
        )

    summary = build_m2_summary(
        prepared_samples=prepared_samples,
        split_manifest=split_manifest,
        source_name=source_name,
        allowed_field_count=len(data_dictionary.allowed_factor_fields),
    )
    return ChatSplitDataset(
        split_records=split_records,
        split_manifest=split_manifest,
        summary=summary,
    )


def assign_report_splits(prepared_samples: list[dict[str, Any]]) -> dict[str, str]:
    report_titles = sorted(
        {sample["report_title"] for sample in prepared_samples if sample.get("report_title")},
        key=_stable_report_sort_key,
    )
    train_count, val_count, test_count = compute_split_counts(len(report_titles))
    assignments: dict[str, str] = {}

    for index, report_title in enumerate(report_titles):
        if index < train_count:
            assignments[report_title] = "train"
        elif index < train_count + val_count:
            assignments[report_title] = "val"
        else:
            assignments[report_title] = "test"
    return assignments


def compute_split_counts(report_count: int) -> tuple[int, int, int]:
    if report_count <= 0:
        return 0, 0, 0
    if report_count == 1:
        return 1, 0, 0
    if report_count == 2:
        return 1, 1, 0

    val_count = max(1, round(report_count * 0.1))
    test_count = max(1, round(report_count * 0.1))
    train_count = report_count - val_count - test_count
    if train_count < 1:
        train_count = 1
        val_count = 1
        test_count = max(0, report_count - train_count - val_count)
    return train_count, val_count, test_count


def build_m2_summary(
    *,
    prepared_samples: list[dict[str, Any]],
    split_manifest: list[dict[str, Any]],
    source_name: str,
    allowed_field_count: int,
) -> dict[str, Any]:
    report_split_counts = {"train": 0, "val": 0, "test": 0}
    sample_split_counts = {"train": 0, "val": 0, "test": 0}
    report_titles_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for item in split_manifest:
        split = item["split"]
        sample_split_counts[split] += 1
        report_titles_by_split[split].add(item["report_title"])

    for split, titles in report_titles_by_split.items():
        report_split_counts[split] = len(titles)

    versions = sorted(
        {
            sample["version"]
            for sample in prepared_samples
            if sample.get("version")
        }
    )
    return {
        "source_name": source_name,
        "input_sample_count": len(prepared_samples),
        "unique_report_count": len({sample["report_title"] for sample in prepared_samples}),
        "allowed_field_count": allowed_field_count,
        "report_split_counts": report_split_counts,
        "sample_split_counts": sample_split_counts,
        "versions": versions,
        "split_strategy": {
            "unit": "report_title",
            "target_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
            "assignment": "stable_hash_order",
        },
        "trainer_framework": "trl",
        "train_record_format": "conversational_prompt_completion",
    }


def _stable_report_sort_key(report_title: str) -> tuple[str, str]:
    digest = hashlib.sha256(report_title.encode("utf-8")).hexdigest()
    return digest, report_title
