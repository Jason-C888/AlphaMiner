from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
