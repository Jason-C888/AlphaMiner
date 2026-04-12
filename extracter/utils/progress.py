from __future__ import annotations

from collections.abc import Iterable, Iterator


try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover
    _tqdm = None


def progress(
    iterable: Iterable,
    *,
    total: int | None = None,
    desc: str | None = None,
) -> Iterable | Iterator:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc)
