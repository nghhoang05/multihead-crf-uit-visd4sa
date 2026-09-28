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
    """Seeds every RNG source this project's training loop actually touches:
    Python's stdlib `random` (used by `src/token_batch_sampler.py::
    TokenBudgetBatchSampler`'s document-order shuffling/weighted resampling),
    NumPy (not used directly by this project's own training code, but some
    HuggingFace/transformers internals reach for it), and BOTH of PyTorch's
    RNG streams -- `torch.manual_seed` alone only seeds the CPU generator;
    `torch.cuda.manual_seed_all` is needed separately for GPU-side randomness
    (dropout masks, weight init on CUDA, etc.), and is skipped when no GPU is
    present rather than raising.

    Call ONCE per run, BEFORE constructing the model (so weight
    initialization is reproducible) and before building the train
    `DataLoader`/`TokenBudgetBatchSampler` (so its shuffle order is
    reproducible) -- see notebook 19's multi-seed reproducibility loop
    (`SEEDS`) for the intended call site. Does not, by itself, guarantee
    bit-exact reproducibility across runs (GPU floating-point reduction
    order and cuDNN's default non-deterministic kernels are separate sources
    of run-to-run variance this project does not otherwise force
    deterministic -- see the multi-seed loop's own docstring/markdown for
    why running several seeds and reporting mean+-std is this project's
    answer to that, rather than chasing bit-exact determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_xlmr_params(model) -> tuple[list, list]:
    """Returns (xlmr_trainable_params, other_trainable_params). Fine-tuning
    XLM-R needs a much smaller learning rate than newly-initialized layers
    (BiLSTM/CRF/CharLSTM/syllable embedding) -- using one shared LR for both
    either destroys the pretrained XLM-R weights in a handful of steps (if
    the shared LR is set for the new layers, e.g. 1e-3) or trains the new
    layers far too slowly (if set for XLM-R, e.g. 2e-5). If XLM-R is frozen
    (`freeze_contextual=True`), its params have requires_grad=False and are
    naturally excluded from both lists already, so this is a no-op in that
    case."""
    contextual_encoder = getattr(getattr(model, "embedding_fusion", None), "contextual_encoder", None)
    xlmr_ids = {id(p) for p in contextual_encoder.xlmr.parameters()} if contextual_encoder is not None else set()

    xlmr_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (xlmr_params if id(p) in xlmr_ids else other_params).append(p)
    return xlmr_params, other_params
