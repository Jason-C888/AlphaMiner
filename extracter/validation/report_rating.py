from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..parser.data_dict_parser import DataDictionary
from ..parser.pdf_parser import PdfParseError, parse_pdf
from ..parser.parser_utils import split_paragraphs
from ..utils.io_utils import FailureRecord
from ..utils.progress import progress


POSITIVE_KEYWORDS = ("因子", "模型", "构建", "定义", "公式", "回归", "打分", "选股")
NEGATIVE_KEYWORDS = ("市场综述", "回测", "业绩", "宏观", "年度策略")
PREFERRED_BROKERS = ("华泰", "海通", "东方", "中信", "广发", "兴业", "长江", "国泰君安", "民生", "东北")


@dataclass(frozen=True)
class CandidateReport:
    report_title: str
    score: float
    rank: int
    report_path: str
    report_date: str | None
    broker: str | None
    text_length: int
    keyword_signal_count: int
    section_signal_count: int
    candidate_section_count: int
    garble_ratio: float


def rate_reports(
    *,
    pdf_paths: list[Path],
    top_k: int,
    data_dictionary: DataDictionary,
) -> tuple[list[CandidateReport], list[FailureRecord]]:
    del data_dictionary
    rated_reports: list[CandidateReport] = []
    failures: list[FailureRecord] = []
    for pdf_path in progress(pdf_paths, total=len(pdf_paths), desc="Discovery"):
        report_date = _extract_report_date(pdf_path.stem)
        broker = _extract_broker(pdf_path.stem, "")
        try:
            parsed = parse_pdf(pdf_path)
        except PdfParseError as exc:
            failures.append(
                FailureRecord(
                    stage="discovery",
                    report_title=pdf_path.stem,
                    reason_type="PARSE_ERROR",
                    reason=str(exc),
                )
            )
            continue

        broker = _extract_broker(pdf_path.stem, parsed.first_page_text)
        score_data = _score_report(parsed.full_text, broker)
        if score_data["text_extractable"] == 0:
            failures.append(
                FailureRecord(
                    stage="discovery",
                    report_title=pdf_path.stem,
                    reason_type="LOW_QUALITY",
                    reason="Report text is empty after extraction.",
                )
            )
            continue
        rated_reports.append(
            CandidateReport(
                report_title=pdf_path.stem,
                score=score_data["score"],
                rank=0,
                report_path=str(pdf_path),
                report_date=report_date,
                broker=broker,
                text_length=score_data["text_length"],
                keyword_signal_count=score_data["keyword_signal_count"],
                section_signal_count=score_data["section_signal_count"],
                candidate_section_count=score_data["candidate_section_count"],
                garble_ratio=score_data["garble_ratio"],
            )
        )

    ranked = sorted(rated_reports, key=lambda item: item.score, reverse=True)[:top_k]
    return (
        [
            CandidateReport(
                report_title=item.report_title,
                score=item.score,
                rank=index,
                report_path=item.report_path,
                report_date=item.report_date,
                broker=item.broker,
                text_length=item.text_length,
                keyword_signal_count=item.keyword_signal_count,
                section_signal_count=item.section_signal_count,
                candidate_section_count=item.candidate_section_count,
                garble_ratio=item.garble_ratio,
            )
            for index, item in enumerate(ranked, start=1)
        ],
        failures,
    )


def _extract_report_date(filename_stem: str) -> str | None:
    match = re.search(r"(20\d{6})", filename_stem)
    if not match:
        return None
    digits = match.group(1)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _extract_broker(filename_stem: str, first_page_text: str) -> str | None:
    search_space = f"{filename_stem} {first_page_text}"
    for broker in PREFERRED_BROKERS:
        if broker in search_space:
            return broker
    return None


def _score_report(text: str, broker: str | None) -> dict[str, float | int]:
    clean_text = text.strip()
    non_space_length = len("".join(clean_text.split()))
    text_extractable = 1 if clean_text else 0
    text_length = len(clean_text)
    section_signal_count = len(re.findall(r"(第[一二三四五六七八九十]+[章节部分])|(^\d+(\.\d+)*)", text, flags=re.MULTILINE))
    keyword_signal_count = sum(text.count(keyword) for keyword in POSITIVE_KEYWORDS)
    garble_ratio = _estimate_garble_ratio(clean_text, non_space_length)
    candidate_section_count = len(discover_candidate_sections(clean_text))
    broker_priority = 1 if broker is not None else 0
    score = (
        text_extractable * 30
        + min(text_length / 1200, 20)
        + min(section_signal_count * 1.5, 15)
        + min(keyword_signal_count * 1.2, 20)
        + min(candidate_section_count * 2.0, 10)
        + broker_priority * 5
        - min(garble_ratio * 100, 20)
    )
    return {
        "score": round(score, 4),
        "text_extractable": text_extractable,
        "text_length": text_length,
        "section_signal_count": section_signal_count,
        "keyword_signal_count": keyword_signal_count,
        "candidate_section_count": candidate_section_count,
        "garble_ratio": round(garble_ratio, 6),
    }


def _count_candidate_sections(text: str) -> int:
    return len(discover_candidate_sections(text))


def discover_candidate_sections(
    text: str,
    *,
    max_sections: int = 8,
    min_length: int = 120,
) -> list[str]:
    scored_sections: list[tuple[int, int, str]] = []
    for paragraph in split_paragraphs(text):
        positive = sum(paragraph.count(keyword) for keyword in POSITIVE_KEYWORDS)
        negative = sum(paragraph.count(keyword) for keyword in NEGATIVE_KEYWORDS)
        score = positive * 2 - negative
        if len(paragraph) < min_length or score <= 0:
            continue
        scored_sections.append((score, len(paragraph), paragraph))
    scored_sections.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [paragraph for _, _, paragraph in scored_sections[:max_sections]]


def _estimate_garble_ratio(text: str, non_space_length: int) -> float:
    if not text or non_space_length == 0:
        return 1.0
    bad_token_count = text.count("\ufffd") + text.lower().count("cid:")
    return bad_token_count / non_space_length
