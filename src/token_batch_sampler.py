"""
Dynamic token-budget batch sampler. Interprets the paper's `batch_size=5000`
(Section 5.1) as a TOKEN budget, not a sentence count -- 5000 sentences per
batch would be infeasible on a single T4 GPU. Sentences are greedily packed
into a batch until adding the next one would exceed the token budget, so
batch composition (sentence count) varies.
"""
from __future__ import annotations

import random

from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Yields lists of dataset indices, each list's total token count close
    to (but not exceeding) `token_budget` -- a single document longer than
    the budget still gets its own one-item batch.

    `shuffle=True` (training) reshuffles document order every epoch before
    packing. `shuffle=False` (dev/test) packs in dataset order for
    deterministic evaluation batches."""

    def __init__(self, lengths: list[int], token_budget: int = 5000, shuffle: bool = True, seed: int = 0):
        if any(n <= 0 for n in lengths):
            raise ValueError("all document lengths must be positive (empty documents should be filtered upstream)")
        self.lengths = lengths
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (before iterating) so the shuffle differs
        across epochs while staying reproducible given the base seed."""
        self._epoch = epoch

    def _order(self) -> list[int]:
        rng = random.Random(self.seed + self._epoch)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        return order

    def __iter__(self):
        batch: list[int] = []
        batch_tokens = 0
        for idx in self._order():
            n = self.lengths[idx]
            if batch and batch_tokens + n > self.token_budget:
                yield batch
                batch, batch_tokens = [], 0
            batch.append(idx)
            batch_tokens += n
        if batch:
            yield batch

    def __len__(self) -> int:
        """Approximate batch count (exact count varies per epoch when
        shuffle=True) -- for progress bars only."""
        total_tokens = sum(self.lengths)
        return max(1, -(-total_tokens // self.token_budget))  # ceil division
