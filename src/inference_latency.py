"""
Measures inference latency of the proposed model, broken into stages
(CUDA-synced before/after each stage for accurate timing):
  - `model_forward_ms`: embedding fusion + BiLSTM + Viterbi decode (2 CRFs)
  - `spans_decode_ms`: tag ids -> spans (twice, once per head)
  - `merge_ms`: `merge_aspect_polarity_spans` -- the merge step's own cost
  - `total_ms`/`docs_per_sec`: sum of the 3 stages above

`n_warmup_batches` excludes the first few batches from timing (XLM-R's
first forward pass pays a one-off kernel-compile cost).
"""
from __future__ import annotations

import time

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _tokens_with_offsets(tokens: list[str], token_offsets: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    return [(tok, start, end) for tok, (start, end) in zip(tokens, token_offsets)]


def measure_latency_multihead(
    model, loader, aspect_tag_vocab, polarity_tag_vocab, device: torch.device,
    n_warmup_batches: int = 2, n_timed_batches: int | None = None, use_amp: bool = True,
) -> dict:
    """Measures latency of the proposed model (2 independent CRF heads),
    timing `spans_decode_ms` and `merge_ms` separately from the model
    forward pass so each stage's cost is visible on its own."""
    from src.multihead_training import merge_aspect_polarity_spans, move_batch_to_device
    from src.span_detection import bio_to_spans

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    batches = list(loader)
    if n_timed_batches is not None:
        batches = batches[: n_warmup_batches + n_timed_batches]

    n_docs = 0
    forward_time = spans_time = merge_time = 0.0

    for bi, batch in enumerate(batches):
        batch_dev = move_batch_to_device(batch, device)
        timed = bi >= n_warmup_batches

        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            aspect_paths, polarity_paths = model.predict(batch_dev)
        _sync(device)
        t1 = time.perf_counter()

        per_doc_spans = []
        for i in range(len(aspect_paths)):
            toks = _tokens_with_offsets(batch["tokens"][i], batch["token_offsets"][i])
            aspect_tags = [aspect_tag_vocab.decode(t) for t in aspect_paths[i]]
            polarity_tags = [polarity_tag_vocab.decode(t) for t in polarity_paths[i]]
            aspect_spans = bio_to_spans(toks, aspect_tags)
            polarity_spans = bio_to_spans(toks, polarity_tags)
            per_doc_spans.append((aspect_spans, polarity_spans))
        _sync(device)
        t2 = time.perf_counter()

        for aspect_spans, polarity_spans in per_doc_spans:
            merge_aspect_polarity_spans(aspect_spans, polarity_spans)
        _sync(device)
        t3 = time.perf_counter()

        if timed:
            forward_time += t1 - t0
            spans_time += t2 - t1
            merge_time += t3 - t2
            n_docs += len(batch["doc_ids"])

    total_time = forward_time + spans_time + merge_time
    return {
        "config": "proposed model (2 independent CRFs, 21+7 tags, WITH merge)",
        "n_docs_timed": n_docs,
        "model_forward_ms_per_doc": forward_time / n_docs * 1000 if n_docs else 0.0,
        "spans_decode_ms_per_doc": spans_time / n_docs * 1000 if n_docs else 0.0,
        "merge_ms_per_doc": merge_time / n_docs * 1000 if n_docs else 0.0,
        "total_ms_per_doc": total_time / n_docs * 1000 if n_docs else 0.0,
        "docs_per_sec": n_docs / total_time if total_time else 0.0,
    }


def print_latency_report(result: dict, log_fn=print) -> None:
    """Prints a per-stage latency breakdown, isolating `merge_ms` (the
    merge step's own cost) instead of only reporting a single total."""
    log_fn(f"{'Stage':<40}{'ms/sentence':>12}")
    for key, label in [
        ("model_forward_ms_per_doc", "Model forward (encoder + 2 CRF decode)"),
        ("spans_decode_ms_per_doc", "Decode 2 tag sequences -> spans"),
        ("merge_ms_per_doc", "Merge 2 CRFs (merge_aspect_polarity_spans)"),
        ("total_ms_per_doc", "TOTAL"),
    ]:
        log_fn(f"{label:<40}{result[key]:>12.3f}")
    log_fn(f"\nThroughput: {result['docs_per_sec']:.2f} sentences/sec "
           f"(measured on {result['n_docs_timed']} sentences, batch_size=1)")
    merge_share = (result["merge_ms_per_doc"] / result["total_ms_per_doc"] * 100) if result["total_ms_per_doc"] else 0.0
    log_fn(f"The merge step accounts for {merge_share:.1f}% of total per-sentence latency.")
