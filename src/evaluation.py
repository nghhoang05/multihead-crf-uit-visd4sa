"""
Exact-span-match evaluation for the span detection task (Nguyen et al. 2021):
a predicted span is correct only if start/end offsets AND label all match
the gold span. Reports Precision/Recall/F1 at micro and macro average, plus
a per-class breakdown, matching the paper's Tables 3/4/5/6.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.span_detection import bio_to_spans


def decode_doc_spans(tokens_with_offsets: list[tuple[str, int, int]], tag_ids: list[int], tag_vocab) -> set[tuple[int, int, str]]:
    """tag_ids -> {(start, end, label), ...} for one document."""
    tags = [tag_vocab.decode(t) for t in tag_ids]
    spans = bio_to_spans(tokens_with_offsets, tags)
    return {(s["start"], s["end"], s["label"]) for s in spans}


def save_raw_predictions(records: list[dict], output_path: str | Path) -> None:
    """Writes one JSON object per line (JSONL) -- the span-level raw
    predictions `evaluate()` alone discards (it only returns aggregated
    TP/FP/FN per class). `records` is the `details` list from
    `src/multihead_training.py::predict_dataset(..., collect_details=True)`."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_predictions(path: str | Path) -> list[dict]:
    """Inverse of `save_raw_predictions`."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(gold_spans_per_doc: list[set], pred_spans_per_doc: list[set]) -> dict:
    """Exact-span-match evaluation over a dataset. `gold_spans_per_doc`/
    `pred_spans_per_doc`: parallel lists, each element a set of
    (start, end, label) tuples for one document (see `decode_doc_spans`).

    Returns micro P/R/F1 (pooled TP/FP/FN across all docs/classes), macro
    P/R/F1 (unweighted mean of per-class F1, over classes with gold
    support > 0), and a per-class breakdown dict."""
    assert len(gold_spans_per_doc) == len(pred_spans_per_doc)

    tp_by_class, fp_by_class, fn_by_class = Counter(), Counter(), Counter()
    for gold, pred in zip(gold_spans_per_doc, pred_spans_per_doc):
        for span in pred & gold:
            tp_by_class[span[2]] += 1
        for span in pred - gold:
            fp_by_class[span[2]] += 1
        for span in gold - pred:
            fn_by_class[span[2]] += 1

    all_classes = sorted(set(tp_by_class) | set(fp_by_class) | set(fn_by_class))
    per_class = {c: _prf(tp_by_class[c], fp_by_class[c], fn_by_class[c]) for c in all_classes}

    micro = _prf(sum(tp_by_class.values()), sum(fp_by_class.values()), sum(fn_by_class.values()))

    gold_classes = [c for c in all_classes if tp_by_class[c] + fn_by_class[c] > 0]
    macro = {
        "precision": sum(per_class[c]["precision"] for c in gold_classes) / len(gold_classes) if gold_classes else 0.0,
        "recall": sum(per_class[c]["recall"] for c in gold_classes) / len(gold_classes) if gold_classes else 0.0,
        "f1": sum(per_class[c]["f1"] for c in gold_classes) / len(gold_classes) if gold_classes else 0.0,
    }

    return {"micro": micro, "macro": macro, "per_class": per_class}
