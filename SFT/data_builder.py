from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "report_title",
    "report_date",
    "broker",
    "inspiration",
    "reasoning",
    "factor_formula",
    "factor_python",
    "required_inputs",
    "inavailable_inputs",
)


@dataclass(frozen=True)
class ReviewResult:
    keep: bool
    issues: list[str]
    implemented: bool


@dataclass(frozen=True)
class ClassificationResult:
    level: str | None
    implemented: bool


@dataclass(frozen=True)
class PreparedDataset:
    prepared_samples: list[dict[str, Any]]
    review_records: list[dict[str, Any]]
    summary: dict[str, Any]


def prepare_dataset(lines: list[str], *, version_id: str, source_name: str) -> PreparedDataset:
    prepared_samples: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    malformed_rows = 0
    schema_issue_rows = 0

    for line_number, raw_line in enumerate(lines, start=1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        try:
            sample = json.loads(stripped_line)
        except json.JSONDecodeError as exc:
            malformed_rows += 1
            review_records.append(
                {
                    "line_number": line_number,
                    "sample_id": None,
                    "report_title": None,
                    "status": "invalid_json",
                    "issues": [f"json_decode_error: {exc.msg}"],
                    "cleaning_implemented": False,
                    "classification_implemented": False,
                }
            )
            continue

        normalized_sample, schema_issues = normalize_sample(sample, version_id=version_id)
        if schema_issues:
            schema_issue_rows += 1

        review_result = review_sample(normalized_sample)
        classification_result = classify_sample(normalized_sample)
        normalized_sample["class"] = classification_result.level

        status = "kept" if review_result.keep else "dropped"
        all_issues = [*schema_issues, *review_result.issues]
        if review_result.keep:
            prepared_samples.append(normalized_sample)

        review_records.append(
            {
                "line_number": line_number,
                "sample_id": normalized_sample["sample_id"],
                "report_title": normalized_sample["report_title"],
                "status": status,
                "issues": all_issues,
                "cleaning_implemented": review_result.implemented,
                "classification_implemented": classification_result.implemented,
                "assigned_class": classification_result.level,
            }
        )

    summary = build_summary(
        prepared_samples=prepared_samples,
        review_records=review_records,
        malformed_rows=malformed_rows,
        schema_issue_rows=schema_issue_rows,
        version_id=version_id,
        source_name=source_name,
    )
    return PreparedDataset(
        prepared_samples=prepared_samples,
        review_records=review_records,
        summary=summary,
    )


def normalize_sample(sample: dict[str, Any], *, version_id: str) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(sample)
    schema_issues: list[str] = []

    for field in REQUIRED_SAMPLE_FIELDS:
        if field not in normalized:
            schema_issues.append(f"missing_field:{field}")

    normalized["sample_id"] = _as_string(normalized.get("sample_id"))
    normalized["report_title"] = _as_string(normalized.get("report_title"))
    normalized["report_date"] = _as_optional_string(normalized.get("report_date"))
    normalized["broker"] = _as_optional_string(normalized.get("broker"))
    normalized["class"] = None
    normalized["version"] = version_id
    normalized["inspiration"] = _as_string(normalized.get("inspiration"))
    normalized["reasoning"] = _as_string(normalized.get("reasoning"))
    normalized["factor_formula"] = _as_string(normalized.get("factor_formula"))
    normalized["factor_python"] = _as_string(normalized.get("factor_python"))
    normalized["required_inputs"] = _as_string_list(
        normalized.get("required_inputs"),
        field_name="required_inputs",
        schema_issues=schema_issues,
    )
    normalized["inavailable_inputs"] = _as_string_list(
        normalized.get("inavailable_inputs"),
        field_name="inavailable_inputs",
        schema_issues=schema_issues,
    )
    normalized["length_input"] = _compute_text_length(normalized["inspiration"])
    normalized["length_output"] = _compute_text_length(
        normalized["reasoning"],
        normalized["factor_formula"],
        normalized["factor_python"],
    )
    return normalized, schema_issues


def review_sample(sample: dict[str, Any]) -> ReviewResult:
    _ = sample
    return ReviewResult(
        keep=True,
        issues=[],
        implemented=False,
    )


def classify_sample(sample: dict[str, Any]) -> ClassificationResult:
    _ = sample
    return ClassificationResult(
        level=None,
        implemented=False,
    )


def build_summary(
    *,
    prepared_samples: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
    malformed_rows: int,
    schema_issue_rows: int,
    version_id: str,
    source_name: str,
) -> dict[str, Any]:
    kept_rows = sum(1 for record in review_records if record["status"] == "kept")
    dropped_rows = sum(1 for record in review_records if record["status"] == "dropped")
    unique_reports = {
        sample["report_title"]
        for sample in prepared_samples
        if sample["report_title"]
    }
    unique_brokers = {
        sample["broker"]
        for sample in prepared_samples
        if sample["broker"]
    }
    input_lengths = [sample["length_input"] for sample in prepared_samples]
    output_lengths = [sample["length_output"] for sample in prepared_samples]
    return {
        "version": version_id,
        "source_name": source_name,
        "total_lines": len(review_records),
        "kept_rows": kept_rows,
        "dropped_rows": dropped_rows,
        "malformed_rows": malformed_rows,
        "schema_issue_rows": schema_issue_rows,
        "prepared_sample_count": len(prepared_samples),
        "unique_report_count": len(unique_reports),
        "unique_broker_count": len(unique_brokers),
        "length_stats": {
            "length_input": _build_length_stats(input_lengths),
            "length_output": _build_length_stats(output_lengths),
        },
        "cleaning_hook": {
            "implemented": False,
            "mode": "placeholder_keep_all",
        },
        "classification_hook": {
            "implemented": False,
            "mode": "placeholder_null_class",
        },
    }


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_string_list(
    value: Any,
    *,
    field_name: str,
    schema_issues: list[str],
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_string(item) for item in value]
    schema_issues.append(f"invalid_type:{field_name}")
    return [_as_string(value)]


def _compute_text_length(*parts: str) -> int:
    return sum(len(part) for part in parts if part)


def _build_length_stats(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {
            "min": None,
            "max": None,
            "avg": None,
        }
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
    }
