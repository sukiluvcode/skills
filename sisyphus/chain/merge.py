"""Merger — optional post-extraction step.

Sits between Extraction and Writer. Receives the full list of `Extracted`
results from one paper and returns a transformed list (dedup, entity
resolution, cross-paragraph roll-up, etc.).

Usage:
    chain = Filter(db) + load + Extraction(...) + Merger(my_merge_fn) + Writer(out_db)
"""
from typing import Callable, Optional

from sisyphus.chain.chain_elements import BaseElement
from sisyphus.chain.extract import Extracted


class Merger(BaseElement):
    """Wraps a user callable that operates on the per-paper Extracted list.

    The callable receives `list[Extracted]` and must return `list[Extracted]`
    (possibly empty). Returning an empty list signals "no result" to the
    downstream chain.
    """

    def __init__(self, fn: Callable[[list[Extracted]], list[Extracted]]):
        self.fn = fn

    def invoke(self, extracted: list[Extracted]) -> Optional[list[Extracted]]:
        if not extracted:
            return None
        merged = self.fn(list(extracted))
        return merged or None
