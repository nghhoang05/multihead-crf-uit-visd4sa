"""
Training loop + decode/evaluate for the CRF multi-head span detection model
(`src/multihead_model.py::BiLSTMMultiHeadCRFTagger`), mirroring
`src/training.py::train_model` (same AdamW/AMP/OOM-recovery infrastructure,
CRF-only weight decay scoping) but adapted for the 2-CRF-head loss/predict
shapes, and for merging the two independently-decoded tag sequences
(aspect, polarity) back into the single (start, end, "ASPECT#POLARITY")
format `src/evaluation.py::evaluate` expects -- the SAME evaluation function
used to score the baseline (`src/training.py`) and Span-ViSD
(`src/span_training.py`), so all 3 configurations in this project's
span-detection ablation are scored identically and directly comparable.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from src.evaluation import decode_doc_spans, evaluate
from src.span_detection import bio_to_spans
from src.training import _split_xlmr_params

_TENSOR_KEYS = ("syllable_ids", "tag_ids_aspect", "tag_ids_polarity", "tag_ids_combined",
                "mask", "char_ids", "char_lengths", "subword_ids", "subword_mask", "alignment")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if k in _TENSOR_KEYS else v) for k, v in batch.items()}


def merge_aspect_polarity_spans(
    aspect_spans: list[dict], polarity_spans: list[dict], default_polarity: str = "POSITIVE",
    return_stats: bool = False,
) -> set[tuple[int, int, str]] | tuple[set[tuple[int, int, str]], dict]:
    """Combine independently-decoded aspect spans (from the aspect CRF head,
    each a dict with start/end/label=ASPECT, `src/span_detection.py::
    bio_to_spans`'s output format) and polarity spans (same shape,
    label=POLARITY) into the final (start, end, "ASPECT#POLARITY") set
    `src/evaluation.py::evaluate` expects for both baseline and Span-ViSD.

    The ASPECT head's boundaries are used as the canonical segmentation
    (aspect identification -- "what is being talked about" -- is this
    task's primary structure; polarity is a property OF an aspect mention,
    not an independent span). For each aspect span, the merged polarity is
    whichever polarity span OVERLAPS it the MOST, by character length (a
    simple, deterministic majority-overlap vote); ties keep whichever was
    seen first. If no polarity span overlaps the aspect span AT ALL (the two
    heads fully disagree on where that span is -- expected to be rare once
    both heads are reasonably trained, since they share the same encoder and
    are trained on the same underlying spans), falls back to
    `default_polarity` (this dataset's majority sentiment class, so an
    aspect span is never silently dropped just because the polarity head
    missed it).

    `return_stats=False` (default, fully backward-compatible -- every
    existing caller, including `scripts/smoke_test_multihead_model.py`'s
    direct equality checks against a plain set, keeps working unmodified):
    when True, ALSO returns a stats dict alongside the SAME merged set
    (merge logic itself is completely unchanged, this only instruments it):
    `total_aspect_spans`/`total_polarity_spans` (counts), `default_assigned`
    (how many aspect spans found NO overlapping polarity span and fell back
    to `default_polarity`), and `orphan_polarity_discarded` (how many
    polarity spans never became any aspect span's best match, so they're
    silently absent from every merged span -- tracked via `id()` on the
    polarity span dicts, since dicts aren't hashable). See
    `aggregate_merge_stats` for the dataset-wide rates these are meant to
    feed into."""
    result = set()
    n_default_assigned = 0
    matched_polarity_ids: set[int] = set()
    for a in aspect_spans:
        best_label, best_overlap, best_span = None, 0, None
        for p in polarity_spans:
            overlap = min(a["end"], p["end"]) - max(a["start"], p["start"])
            if overlap > best_overlap:
                best_overlap, best_label, best_span = overlap, p["label"], p
        if best_label is None:
            n_default_assigned += 1
            polarity = default_polarity
        else:
            polarity = best_label
            matched_polarity_ids.add(id(best_span))
        result.add((a["start"], a["end"], f"{a['label']}#{polarity}"))

    if not return_stats:
        return result

    stats = {
        "total_aspect_spans": len(aspect_spans),
        "default_assigned": n_default_assigned,
        "total_polarity_spans": len(polarity_spans),
        "orphan_polarity_discarded": len(polarity_spans) - len(matched_polarity_ids),
    }
    return result, stats


def aggregate_merge_stats(all_stats: list[dict]) -> dict:
    """Cộng dồn stats trả về bởi `merge_aspect_polarity_spans(...,
    return_stats=True)` qua toàn bộ tập test (1 dict/câu) thành 1 bộ thống
    kê tổng, và tính 2 tỉ lệ báo cáo: % span aspect phải nhận cực tính MẶC
    ĐỊNH (không polarity span nào chồng lấn -- dấu hiệu 2 head bất đồng ranh
    giới), và % polarity span "đơn độc" bị loại khỏi kết quả cuối (chưa từng
    là lựa chọn tốt nhất cho aspect span nào)."""
    total_aspect = sum(s["total_aspect_spans"] for s in all_stats)
    total_default = sum(s["default_assigned"] for s in all_stats)
    total_polarity = sum(s["total_polarity_spans"] for s in all_stats)
    total_orphan = sum(s["orphan_polarity_discarded"] for s in all_stats)

    default_rate = (total_default / total_aspect * 100) if total_aspect else 0.0
    orphan_rate = (total_orphan / total_polarity * 100) if total_polarity else 0.0

    return {
        "total_aspect_spans": total_aspect, "default_assigned": total_default,
        "default_assigned_rate_pct": default_rate,
        "total_polarity_spans": total_polarity, "orphan_polarity_discarded": total_orphan,
        "orphan_polarity_discarded_rate_pct": orphan_rate,
    }


def _tokens_with_offsets(tokens: list[str], token_offsets: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    return [(tok, start, end) for tok, (start, end) in zip(tokens, token_offsets)]


@torch.no_grad()
def predict_dataset(model, loader, vocab: dict, device: torch.device, use_amp: bool = True, log_fn=print,
                     collect_details: bool = False):
    """Returns (gold_spans_per_doc, pred_spans_per_doc, doc_ids), each a set
    of (start, end, "ASPECT#POLARITY") -- directly consumable by
    `src/evaluation.py::evaluate`, identical format to the baseline and
    Span-ViSD. Gold spans are decoded from `tag_ids_combined` (the SAME
    combined-scheme tags the baseline trains on) via `src/evaluation.py::
    decode_doc_spans` -- the EXACT SAME function `src/training.py::
    predict_dataset` uses for the baseline's own gold-span decoding, so gold
    is identical across all 3 configurations both in DATA and in DECODE
    LOGIC (previously this function re-implemented that same 2-line decode
    inline instead of calling it -- switched to avoid 2 independently-
    maintained copies of identical logic). Only the PREDICTION side differs
    (merged from the 2 independently-decoded heads via
    `merge_aspect_polarity_spans`, which still needs `bio_to_spans`'s
    dict-shaped output directly, not `decode_doc_spans`'s flattened set).

    `collect_details=True` (default False, fully backward-compatible --
    every existing caller, including `train_model`'s own internal dev-set
    calls, keeps getting the exact same 3-tuple) ADDITIONALLY returns a 4th
    list, `details`: one JSON-serializable dict per document with
    `aspect_spans_pred`/`polarity_spans_pred` (the 2 heads' OWN decoded
    spans, before merging), `merged_spans_pred`/`gold_spans` (span-level,
    same shape `src/training.py::predict_dataset`'s `details` uses), and
    this document's own `merge_stats` (from `merge_aspect_polarity_spans(...,
    return_stats=True)` -- feed the per-doc `merge_stats` values into
    `aggregate_merge_stats` for the dataset-wide rates). Costs a little
    extra work per document (the merge stats bookkeeping) but no extra
    forward passes -- purely a post-processing addition over spans the model
    already produced."""
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    aspect_tag_vocab = vocab["tag"]["aspect"]
    polarity_tag_vocab = vocab["tag"]["polarity"]
    combined_tag_vocab = vocab["tag"]["aspect_polarity"]
    gold_all, pred_all, doc_ids_all = [], [], []
    details_all = [] if collect_details else None
    n_oom_skipped = 0
    for batch in loader:
        batch_dev = move_batch_to_device(batch, device)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                aspect_paths, polarity_paths = model.predict(batch_dev)
        except torch.cuda.OutOfMemoryError:
            n_oom_skipped += len(batch["doc_ids"])
            del batch_dev
            torch.cuda.empty_cache()
            continue
        for i in range(len(aspect_paths)):
            n = batch["mask"][i].sum().item()
            toks = _tokens_with_offsets(batch["tokens"][i], batch["token_offsets"][i])

            gold_spans = decode_doc_spans(toks, batch["tag_ids_combined"][i, :n].tolist(), combined_tag_vocab)

            aspect_tags = [aspect_tag_vocab.decode(t) for t in aspect_paths[i]]
            polarity_tags = [polarity_tag_vocab.decode(t) for t in polarity_paths[i]]
            aspect_spans = bio_to_spans(toks, aspect_tags)
            polarity_spans = bio_to_spans(toks, polarity_tags)

            if collect_details:
                pred_spans, merge_stats = merge_aspect_polarity_spans(aspect_spans, polarity_spans, return_stats=True)
            else:
                pred_spans = merge_aspect_polarity_spans(aspect_spans, polarity_spans)

            gold_all.append(gold_spans)
            pred_all.append(pred_spans)
            doc_ids_all.append(batch["doc_ids"][i])

            if collect_details:
                details_all.append({
                    "sentence_id": batch["doc_ids"][i],
                    "tokens": batch["tokens"][i],
                    "aspect_spans_pred": aspect_spans,
                    "polarity_spans_pred": polarity_spans,
                    "merged_spans_pred": sorted(list(s) for s in pred_spans),
                    "gold_spans": sorted(list(s) for s in gold_spans),
                    "merge_stats": merge_stats,
                })
    if n_oom_skipped:
        log_fn(f"  !! CUDA OOM: {n_oom_skipped} tài liệu bị bỏ qua khỏi lần đánh giá này (không tính vào P/R/F1).")
    if collect_details:
        return gold_all, pred_all, doc_ids_all, details_all
    return gold_all, pred_all, doc_ids_all


def _build_multihead_param_groups(model, lr: float, xlmr_lr: float, weight_decay: float) -> list[dict]:
    """Same scoping rationale as `src/training.py::_build_param_groups`
    (Eq. 11's L2 term is specific to the CRF's own parameters) -- adapted
    for TWO CRFs (`crf_aspect`, `crf_polarity`) instead of one."""
    xlmr_params, other_params = _split_xlmr_params(model)
    crf_ids = {id(p) for p in model.crf_aspect.parameters()} | {id(p) for p in model.crf_polarity.parameters()}
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
    vocab: dict,
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
    """Same shape, defaults, and rationale as `src/training.py::train_model`
    (this configuration keeps the baseline's own CRF hyperparameters, unlike
    Span-ViSD, since the extraction mechanism -- CRF decoding -- is
    unchanged here; only the tag-space decomposition differs)."""
    model.to(device)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    param_groups = _build_multihead_param_groups(model, lr=lr, xlmr_lr=xlmr_lr, weight_decay=weight_decay)
    trainable_params = [p for group in param_groups for p in group["params"]]
    optimizer = torch.optim.AdamW(param_groups)
    log_fn(f"Mixed precision (fp16): {'BẬT' if amp_enabled else 'tắt (không phải CUDA hoặc use_amp=False)'}"
           f" | weight_decay={weight_decay} chỉ áp cho 2 CRF (Eq. 11 bài báo), phần còn lại weight_decay=0")

    history = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

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
                scaler.unscale_(optimizer)
                unscale_called = True
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_batches += 1
            except torch.cuda.OutOfMemoryError:
                n_oom_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if unscale_called:
                    scaler.update()  # see src/training.py's identical fix -- resets GradScaler's stuck UNSCALED state
                del batch
                torch.cuda.empty_cache()
                log_fn(f"  !! CUDA OOM ở 1 batch (epoch {epoch}), đã bỏ qua batch này và tiếp tục.")
        train_loss = total_loss / max(n_batches, 1)
        train_time = time.time() - t0
        if n_oom_skipped:
            log_fn(f"  -> Tổng {n_oom_skipped} batch bị bỏ qua vì OOM trong epoch {epoch}.")

        t0 = time.time()
        gold, pred, _ = predict_dataset(model, dev_loader, vocab, device, use_amp=use_amp, log_fn=log_fn)
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
