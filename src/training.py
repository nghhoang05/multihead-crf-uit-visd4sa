"""
Shared training utilities used by `src/multihead_training.py::train_model`:
RNG seeding and XLM-R/non-XLM-R parameter splitting (fine-tuning XLM-R needs
a much smaller learning rate than the rest of the model).
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python's `random`, NumPy, and both PyTorch RNG streams
    (`torch.manual_seed` for CPU, `torch.cuda.manual_seed_all` for GPU).
    Call ONCE per run, before constructing the model and before building the
    train DataLoader/TokenBudgetBatchSampler, so both weight init and batch
    shuffle order are reproducible. Does not guarantee bit-exact
    reproducibility across runs (GPU/cuDNN non-determinism is a separate
    source of run-to-run variance) -- run several seeds and report mean+-std
    instead of chasing bit-exact determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_xlmr_params(model) -> tuple[list, list]:
    """Returns (xlmr_trainable_params, other_trainable_params). Fine-tuning
    XLM-R needs a much smaller learning rate than newly-initialized layers,
    hence the split into separate optimizer param groups. If XLM-R is
    frozen, its params have requires_grad=False and are already excluded."""
    contextual_encoder = getattr(getattr(model, "embedding_fusion", None), "contextual_encoder", None)
    xlmr_ids = {id(p) for p in contextual_encoder.xlmr.parameters()} if contextual_encoder is not None else set()

    xlmr_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (xlmr_params if id(p) in xlmr_ids else other_params).append(p)
    return xlmr_params, other_params
