"""
Device-aware training loop for BiLSTMCRFTagger (works unchanged on CPU or
CUDA -- e.g. this project's local CPU-only machine, or a Google Colab T4
runtime). Early-stops on dev macro-F1 (exact span match, see
`src/evaluation.py`), matching the paper's evaluation protocol.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch

from src.evaluation import decode_doc_spans, evaluate
from src.span_detection import bio_to_spans

_TENSOR_KEYS = ("syllable_ids", "tag_ids", "mask", "char_ids", "char_lengths",
                "subword_ids", "subword_mask", "alignment",
                "phobert_ids", "phobert_mask", "phobert_alignment")


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
    reproducible) -- see notebooks 16/19's multi-seed reproducibility loop
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


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if k in _TENSOR_KEYS else v) for k, v in batch.items()}


@torch.no_grad()
def predict_dataset(model, loader, vocab, scheme: str, device: torch.device, use_amp: bool = True, log_fn=print,
                     collect_details: bool = False):
    """Run the model over an entire DataLoader and decode predictions + gold
    into per-document span sets. Returns (gold_spans_per_doc, pred_spans_per_doc, doc_ids).

    `use_amp=True` runs the forward pass in fp16 autocast on CUDA (no-op on
    CPU) -- cuts memory/time for XLM-R's forward pass during dev/test
    evaluation too, not just training.

    A batch that hits CUDA OOM is skipped (its documents are simply absent
    from the returned lists, logged via `log_fn`) rather than crashing dev/
    test evaluation entirely -- same rationale as `train_model`'s per-batch
    OOM handling. Rare in practice (no backward pass here, so peak memory is
    much lower than training), but the token-budget sampler doesn't bound
    per-sequence length, so it's not impossible.

    `collect_details=True` (default False, fully backward-compatible --
    every existing caller, including `train_model`'s own internal dev-set
    calls, keeps getting the exact same 3-tuple) ADDITIONALLY returns a 4th
    list, `details`: one JSON-serializable dict per document
    (`sentence_id`/`tokens`/`spans_pred`/`gold_spans`) with the RAW span-
    level prediction `evaluate()` aggregates away into TP/FP/FN -- needed
    for error analysis and `scripts/bootstrap_compare_macro_f1.py`. Pass the
    result straight to `src/evaluation.py::save_raw_predictions`."""
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    tag_vocab = vocab["tag"][scheme]
    gold_all, pred_all, doc_ids_all = [], [], []
    details_all = [] if collect_details else None
    n_oom_skipped = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                pred_paths = model.predict(batch)
        except torch.cuda.OutOfMemoryError:
            n_oom_skipped += len(batch["doc_ids"])
            del batch
            torch.cuda.empty_cache()
            continue
        for i in range(len(pred_paths)):
            n = batch["mask"][i].sum().item()
            gold_tag_ids = batch["tag_ids"][i, :n].tolist()
            toks_with_offsets = list(zip(batch["tokens"][i], *zip(*batch["token_offsets"][i])))
            gold_spans = decode_doc_spans(toks_with_offsets, gold_tag_ids, tag_vocab)
            pred_spans = decode_doc_spans(toks_with_offsets, pred_paths[i], tag_vocab)
            gold_all.append(gold_spans)
            pred_all.append(pred_spans)
            doc_ids_all.append(batch["doc_ids"][i])
            if collect_details:
                details_all.append({
                    "sentence_id": batch["doc_ids"][i],
                    "tokens": batch["tokens"][i],
                    "spans_pred": sorted(list(s) for s in pred_spans),
                    "gold_spans": sorted(list(s) for s in gold_spans),
                })
    if n_oom_skipped:
        log_fn(f"  !! CUDA OOM: {n_oom_skipped} tài liệu bị bỏ qua khỏi lần đánh giá này "
                f"(không tính vào P/R/F1) -- xem xét giảm TOKEN_BUDGET nếu số này đáng kể.")
    if collect_details:
        return gold_all, pred_all, doc_ids_all, details_all
    return gold_all, pred_all, doc_ids_all


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


def _build_param_groups(model, lr: float, xlmr_lr: float, weight_decay: float) -> list[dict]:
    """Splits trainable params into up to 3 AdamW param groups:
      - CRF's own parameters (transitions/start/end): lr, weight_decay=weight_decay
      - everything else EXCEPT XLM-R (syllable/char/BiLSTM/Linear/[projection]): lr, weight_decay=0
      - XLM-R's own parameters (if fine-tuning): xlmr_lr, weight_decay=0

    Why weight_decay is scoped to ONLY the CRF: the paper's Eq. (11), Section
    4.3 "Conditional Random Fields (CRF)", writes the CRF's own training
    objective as log-likelihood minus an L2 penalty on lambda_k -- this is
    classic linear-chain CRF theory (Lafferty et al., 2001) describing
    regularization of the CRF's OWN feature-weight parameters, not a
    network-wide weight decay. An earlier version of this code applied
    weight_decay uniformly to every trainable parameter (including XLM-R
    and the embeddings) -- that overreads Eq. (11), which appears specifically
    in the CRF subsection and never claims to cover the embedding/BiLSTM
    layers described in Sections 4.1/4.2 (those only get the stated 0.33
    dropout, no L2 term mentioned anywhere for them)."""
    xlmr_params, other_params = _split_xlmr_params(model)
    crf_ids = {id(p) for p in getattr(model, "crf", torch.nn.Module()).parameters()}
    crf_params = [p for p in other_params if id(p) in crf_ids]
    non_crf_params = [p for p in other_params if id(p) not in crf_ids]

    groups = []
    if non_crf_params:
        groups.append({"params": non_crf_params, "lr": lr, "weight_decay": 0.0})
    if crf_params:
        groups.append({"params": crf_params, "lr": lr, "weight_decay": weight_decay})
    if xlmr_params:
        groups.append({"params": xlmr_params, "lr": xlmr_lr, "weight_decay": 0.0})
    return groups


def train_model(
    model,
    train_loader,
    dev_loader,
    vocab,
    scheme: str,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    xlmr_lr: float = 2e-5,
    weight_decay: float = 1e-2,
    patience: int = 3,
    grad_clip: float = 5.0,
    checkpoint_path: Path | str | None = None,
    use_amp: bool = True,
    log_fn=print,
) -> dict:
    """
    Trains with AdamW over the trainable parameters (any frozen component --
    XLM-R and/or the syllable embedding -- is excluded automatically since
    its params have requires_grad=False), gradient clipping (CRF losses can
    spike early in training), and early stopping on dev macro-F1 (exact span
    match). Saves the best checkpoint's state_dict to `checkpoint_path` if
    given. Returns a history dict for plotting/reporting.

    `xlmr_lr` is a SEPARATE, much smaller learning rate for XLM-R's own
    parameters when it's being fine-tuned (freeze_contextual=False) --
    standard practice for fine-tuning a large pretrained transformer
    alongside newly-initialized layers. Has no effect if XLM-R is frozen or
    `use_contextual=False` (there are no XLM-R params to split out).

    `weight_decay` (default 1e-2): the paper's CRF training objective, Eq.
    (11) in Section 4.3, includes an explicit L2 penalty term (-Sum lambda^2
    / 2*sigma^2) on the CRF's OWN parameters -- no concrete sigma^2 is given
    anywhere in the paper, so this is a reasonable standard AdamW value, not
    a reproduction of a specific paper number. Applied ONLY to `model.crf`'s
    parameters (see `_build_param_groups`), not network-wide -- Eq. (11) is
    specific to the CRF subsection and the paper states no L2 term for the
    embedding/BiLSTM layers (Sections 4.1/4.2), only their 0.33 dropout.
    Uses AdamW (decoupled weight decay) rather than Adam+L2 since the two
    are not equivalent under adaptive learning rates (Loshchilov & Hutter,
    2019) -- AdamW is the more correct implementation of "add an L2-style
    penalty" in this optimizer family.

    `use_amp=True` (default) trains in fp16 mixed precision on CUDA (auto
    no-op on CPU, so this stays safe for this project's local CPU-only
    machine too) -- combined with `ContextualEncoder`'s gradient
    checkpointing, this is what makes fine-tuning XLM-R fit in a free-tier
    T4's 16GB after it previously hit `CUDA out of memory` at fp32.
    """
    model.to(device)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    param_groups = _build_param_groups(model, lr=lr, xlmr_lr=xlmr_lr, weight_decay=weight_decay)
    trainable_params = [p for group in param_groups for p in group["params"]]
    xlmr_params, _ = _split_xlmr_params(model)
    if xlmr_params:
        log_fn(f"XLM-R đang fine-tune: {sum(p.numel() for p in xlmr_params):,} tham số, lr={xlmr_lr} "
               f"(riêng với lr={lr} cho {sum(p.numel() for p in trainable_params) - sum(p.numel() for p in xlmr_params):,} tham số còn lại)")
    optimizer = torch.optim.AdamW(param_groups)
    log_fn(f"Mixed precision (fp16): {'BẬT' if amp_enabled else 'tắt (không phải CUDA hoặc use_amp=False)'}"
           f" | weight_decay={weight_decay} chỉ áp cho CRF (Eq. 11 bài báo), phần còn lại weight_decay=0")

    history = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)  # reshuffle TokenBudgetBatchSampler's packing each epoch

        model.train()
        t0 = time.time()
        total_loss, n_batches, n_oom_skipped = 0.0, 0, 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            unscale_called = False
            try:
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    loss = model.loss(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # so grad_clip sees true-scale gradients, not the fp16 loss-scaled ones
                unscale_called = True
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_batches += 1
            except torch.cuda.OutOfMemoryError:
                # A single unusually large batch (TokenBudgetBatchSampler bounds total
                # tokens, not max sequence length or sequence count, so an occasional
                # spike is possible) shouldn't kill an otherwise-fine training run --
                # drop this one batch and keep going, instead of crashing the whole
                # epoch. Recorded in `history` so this is visible in the results file,
                # not silently swallowed.
                n_oom_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if unscale_called:
                    # scaler.unscale_(optimizer) already ran before the OOM (e.g. it hit
                    # during clip_grad_norm_ or scaler.step() instead of backward()) --
                    # GradScaler's per-optimizer state is now stuck at UNSCALED until a
                    # matching update() call resets it, or EVERY subsequent unscale_()
                    # this run raises "unscale_() has already been called ... since the
                    # last update()". update() always clears that state (its last line
                    # resets `_per_optimizer_states`), so call it here to recover -- but
                    # ONLY when unscale_ actually ran, since update() otherwise asserts
                    # "No inf checks were recorded prior to update.".
                    scaler.update()
                del batch
                torch.cuda.empty_cache()
                log_fn(f"  !! CUDA OOM ở 1 batch (epoch {epoch}), đã bỏ qua batch này và tiếp tục.")
        train_loss = total_loss / max(n_batches, 1)
        train_time = time.time() - t0
        if n_oom_skipped:
            log_fn(f"  -> Tổng {n_oom_skipped} batch bị bỏ qua vì OOM trong epoch {epoch} "
                    f"(train_loss chỉ tính trên {n_batches} batch còn lại).")

        t0 = time.time()
        gold, pred, _ = predict_dataset(model, dev_loader, vocab, scheme, device, use_amp=use_amp, log_fn=log_fn)
        dev_metrics = evaluate(gold, pred)
        eval_time = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": train_loss,
            "dev_micro_f1": dev_metrics["micro"]["f1"], "dev_macro_f1": dev_metrics["macro"]["f1"],
            "dev_micro_precision": dev_metrics["micro"]["precision"], "dev_micro_recall": dev_metrics["micro"]["recall"],
            "train_time_sec": train_time, "eval_time_sec": eval_time, "n_oom_skipped": n_oom_skipped,
        }
        history.append(record)
        log_fn(f"[epoch {epoch}/{epochs}] loss={train_loss:.3f} "
               f"dev_micro_F1={dev_metrics['micro']['f1']:.4f} dev_macro_F1={dev_metrics['macro']['f1']:.4f} "
               f"(train {train_time:.0f}s, eval {eval_time:.0f}s)")

        if dev_metrics["macro"]["f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro"]["f1"]
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
                log_fn(f"  -> dev macro-F1 cải thiện, đã lưu checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log_fn(f"  -> dừng sớm (early stopping): {patience} epoch liên tiếp không cải thiện dev macro-F1")
                break

    return {"history": history, "best_dev_macro_f1": best_macro_f1}
