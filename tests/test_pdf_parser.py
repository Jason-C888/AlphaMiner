from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extracter.parser.pdf_parser import PdfParseError, parse_pdf


def test_parse_pdf_wraps_page_enumeration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePages:
        def __iter__(self):
            raise RuntimeError("PyCryptodome is required for AES algorithm")

    class FakeReader:
        def __init__(self, _: str) -> None:
            self.pages = FakePages()

    monkeypatch.setitem(sys.modules, "PyPDF2", types.SimpleNamespace(PdfReader=FakeReader))

    with pytest.raises(PdfParseError, match="Failed to read PDF pages"):
        parse_pdf(Path("encrypted.pdf"))
