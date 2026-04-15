from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from .parser_utils import normalize_text


class PdfParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedPdf:
    path: Path
    page_count: int
    first_page_text: str
    full_text: str


def is_pdf_parser_available() -> bool:
    try:
        from PyPDF2 import PdfReader  # noqa: F401
    except ModuleNotFoundError:
        _inject_vendor_path()
        try:
            from PyPDF2 import PdfReader  # noqa: F401
        except ModuleNotFoundError:
            return False
    return True


def parse_pdf(path: str | Path) -> ParsedPdf:
    pdf_path = Path(path)
    try:
        from PyPDF2 import PdfReader
    except ModuleNotFoundError as exc:
        _inject_vendor_path()
        try:
            from PyPDF2 import PdfReader
        except ModuleNotFoundError as retry_exc:
            raise PdfParseError("PyPDF2 is not installed.") from retry_exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        raise PdfParseError(f"Failed to open PDF: {exc}") from exc

    try:
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise PdfParseError(f"Failed to read PDF pages: {exc}") from exc
    if not pages:
        raise PdfParseError("PDF contains no readable pages.")
    pages_to_parse = pages[:-1] if len(pages) > 1 else pages

    page_texts: list[str] = []
    for page in pages_to_parse:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            raise PdfParseError(f"Failed to extract PDF text: {exc}") from exc
        page_texts.append(page_text)

    first_page_text = normalize_text(page_texts[0])
    full_text = normalize_text("\n".join(page_texts))
    if not full_text:
        raise PdfParseError("PDF text extraction returned empty content.")

    return ParsedPdf(
        path=pdf_path,
        page_count=len(page_texts),
        first_page_text=first_page_text,
        full_text=full_text,
    )


def _inject_vendor_path() -> None:
    vendor_path = Path(__file__).resolve().parent.parent / ".vendor"
    vendor_path_str = str(vendor_path)
    if vendor_path.exists() and vendor_path_str not in sys.path:
        sys.path.insert(0, vendor_path_str)
