from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm as _tqdm


def iter_progress(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
):
    return _tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
    )


class StageProgress:
    def __init__(self, progress_bar: Any):
        self._progress_bar = progress_bar

    def advance(self, message: str, amount: int = 1) -> None:
        self._progress_bar.set_postfix_str(message, refresh=False)
        self._progress_bar.update(amount)


@contextmanager
def stage_progress(*, total: int, desc: str, unit: str = "step") -> Iterator[StageProgress]:
    progress_bar = _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    try:
        yield StageProgress(progress_bar)
    finally:
        progress_bar.close()
