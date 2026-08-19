"""
Dynamic token-budget batch sampler, replacing the previous fixed
`batch_size=32 sentences` used in the earlier training runs (see
`reports/span_detection/results_aspect_polarity_colab.json`).

Interprets the paper's `batch_size=5000` (Section 5.1) as a TOKEN budget,
not a sentence count -- 5000 SENTENCES per batch would be infeasible on a
single T4 GPU even with a frozen XLM-R (see `files/README.md`'s own note on
this). Sentences are greedily packed into a batch until adding the next one
would exceed the token budget, so batch composition (sentence count) varies:
many short reviews per batch, few for long ones.

This changes how many optimizer steps happen per epoch compared to the old
fixed-batch-size runs, so results from before this change are not directly
comparable -- retrain from scratch under this sampler for consistent numbers.
"""
from __future__ import annotations

import random

from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Yields lists of dataset indices, each list's total token count close
    to (but not exceeding) `token_budget` -- except a single document longer
    than the budget still gets its own one-item batch rather than being
    dropped or truncated.

    `shuffle=True` (use for training) reshuffles document order every epoch
    before packing, so batch composition varies across epochs. `shuffle=False`
    (use for dev/test) packs in dataset order for deterministic, reproducible
    evaluation batches.

    `weights` (optional, default `None` -- preserves every existing caller's
    behavior exactly, this parameter is purely additive): per-document
    sampling weights for WEIGHTED resampling WITH replacement (Đề xuất "Multi-
    head hoàn chỉnh giải quyết lớp hiếm", Thành phần D -- see `src/
    rare_class_sampling.py::compute_sentence_sampling_weights`), drawing
    exactly `len(lengths)` indices per epoch the same way `torch.utils.data.
    WeightedRandomSampler(replacement=True)` would, then packing that
    (possibly-duplicated) index sequence into token-budget batches with the
    SAME greedy-packing logic used for the plain shuffle case -- so documents
    about rare aspects/NEUTRAL polarity can appear more than once per epoch
    (and common ones less than once, on average) without needing a second,
    incompatible sampler class alongside this one. When given, `shuffle` is
    ignored (weighted-with-replacement sampling already reorders every
    epoch); `weights` must be the same length as `lengths`, in the same
    (unshuffled) index order.
    """

    def __init__(
        self, lengths: list[int], token_budget: int = 5000, shuffle: bool = True, seed: int = 0,
        weights: list[float] | None = None,
    ):
        if any(n <= 0 for n in lengths):
            raise ValueError("all document lengths must be positive (empty documents should be filtered upstream)")
        if weights is not None and len(weights) != len(lengths):
            raise ValueError(f"weights length ({len(weights)}) must match lengths length ({len(lengths)})")
        self.lengths = lengths
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = seed
        self.weights = weights
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (before iterating) so the shuffle differs
        across epochs while staying reproducible given the base seed."""
        self._epoch = epoch

    def _order(self) -> list[int]:
        rng = random.Random(self.seed + self._epoch)
        n = len(self.lengths)
        if self.weights is not None:
            return rng.choices(range(n), weights=self.weights, k=n)
        order = list(range(n))
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
        """Approximate batch count (exact count depends on packing order,
        which varies per epoch when shuffle=True) -- for progress bars only."""
        total_tokens = sum(self.lengths)
        return max(1, -(-total_tokens // self.token_budget))  # ceil division
