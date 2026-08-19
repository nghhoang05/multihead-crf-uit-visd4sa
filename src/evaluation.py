"""
Exact-span-match evaluation for the span detection task (Nguyen et al. 2021):
"A predicted span is correct only if it exactly matches the gold standard
span" -- both start/end offsets AND label must match. Reports Precision/
Recall/F1 at both micro and macro average, plus a per-class breakdown table,
matching the paper's Tables 3/4/5/6.

Also provides `evaluate_relaxed` (NOT part of the paper's own stated
methodology -- an explicit, deliberate deviation added to test a specific
hypothesis): an OVERLAP-match variant that drops the exact-boundary
requirement entirely and only checks that a predicted span's LABEL matches a
gold span it overlaps with at all. Added after several real, confirmed
boundary/offset bugs were found and fixed in this reproduction (see
`src/hfs_preprocess.py`, `src/hfs_aspect_dataset.py`) plus a structural,
unfixable residual (PhoBERT compressing a whole multi-syllable word into one
BPE token, see `HfsAspectCollator`) -- all of which push exact-match F1 down
for reasons that have nothing to do with whether the model identified the
right ASPECT (or ASPECT#POLARITY) in roughly the right place. `evaluate_
relaxed` isolates "does the model get the category right, independent of
nailing the exact character boundary" from "does it also nail the exact
boundary" (`evaluate`) -- comparing the two numbers directly answers that
question instead of guessing at it.
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
    prediction detail `evaluate()` alone doesn't preserve (it only returns
    already-aggregated TP/FP/FN per class, discarding which SENTENCE each
    one came from). `records` is whatever list of per-document dicts `src/
    training.py::predict_dataset` or `src/multihead_training.py::
    predict_dataset` built when called with `collect_details=True` -- this
    function doesn't interpret or validate their shape, just serializes
    them, one line each, so downstream tools (error analysis, `scripts/
    bootstrap_compare_macro_f1.py`) can stream the file instead of loading
    one giant JSON array."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_predictions(path: str | Path) -> list[dict]:
    """Inverse of `save_raw_predictions` -- reads a JSONL file back into a
    list of dicts, in the same order they were written."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _aggregate(tp_by_class: Counter, fp_by_class: Counter, fn_by_class: Counter) -> dict:
    """Shared micro/macro/per-class rollup for both `evaluate` and
    `evaluate_relaxed` -- identical aggregation, they only differ in HOW a
    single (gold, pred) pair is counted as tp/fp/fn in the first place."""
    all_classes = sorted(set(tp_by_class) | set(fp_by_class) | set(fn_by_class))
    per_class = {
        c: _prf(tp_by_class[c], fp_by_class[c], fn_by_class[c])
        for c in all_classes
    }

    micro = _prf(sum(tp_by_class.values()), sum(fp_by_class.values()), sum(fn_by_class.values()))

    gold_classes = [c for c in all_classes if tp_by_class[c] + fn_by_class[c] > 0]
    macro = {
        "precision": sum(per_class[c]["precision"] for c in gold_classes) / len(gold_classes) if gold_classes else 0.0,
        "recall": sum(per_class[c]["recall"] for c in gold_classes) / len(gold_classes) if gold_classes else 0.0,
        "f1": sum(per_class[c]["f1"] for c in gold_classes) / len(gold_classes) if gold_classes else 0.0,
    }

    return {"micro": micro, "macro": macro, "per_class": per_class}


def evaluate(gold_spans_per_doc: list[set], pred_spans_per_doc: list[set]) -> dict:
    """
    Exact-span-match evaluation over a dataset.
    gold_spans_per_doc / pred_spans_per_doc: parallel lists, each element a
    set of (start, end, label) tuples for one document (see `decode_doc_spans`).

    Returns micro P/R/F1 (pooled TP/FP/FN across all docs and classes), macro
    P/R/F1 (unweighted mean of per-class F1, over classes with gold support > 0
    -- classes never mentioned in gold are excluded, matching standard macro-F1
    practice), and a per-class breakdown dict.
    """
    assert len(gold_spans_per_doc) == len(pred_spans_per_doc)

    tp_by_class, fp_by_class, fn_by_class = Counter(), Counter(), Counter()
    for gold, pred in zip(gold_spans_per_doc, pred_spans_per_doc):
        for span in pred & gold:
            tp_by_class[span[2]] += 1
        for span in pred - gold:
            fp_by_class[span[2]] += 1
        for span in gold - pred:
            fn_by_class[span[2]] += 1

    return _aggregate(tp_by_class, fp_by_class, fn_by_class)


def evaluate_relaxed(gold_spans_per_doc: list[set], pred_spans_per_doc: list[set]) -> dict:
    """
    OVERLAP-match evaluation -- same P/R/F1/micro/macro/per-class shape as
    `evaluate`, but a gold span and a predicted span count as a match
    whenever (a) their LABEL is identical (the full label as given -- pass
    aspect-only labels for an aspect-only comparison, "ASPECT#POLARITY"
    labels for a combined comparison; this function does not split or
    otherwise interpret the label string) and (b) their `[start, end)`
    character ranges overlap AT ALL (share at least one character) --
    `start`/`end` no longer need to match exactly.

    Matching is ONE-TO-ONE per document (greedy, largest character-overlap
    first): each gold span can satisfy at most one predicted span and vice
    versa, so one long/fragmented prediction cannot inflate TP by "covering"
    several distinct gold spans (or the reverse) -- a real risk once exact
    boundaries stop being required. Any gold span left unmatched after
    greedy assignment is a FN; any predicted span left unmatched is a FP.

    Intentionally simpler than IoU-thresholded matching (no minimum overlap
    fraction) -- any nonzero character overlap counts, matching this
    function's specific purpose (isolate "wrong category" from "right
    category, imperfect boundary"), not a general-purpose partial-credit
    metric.
    """
    assert len(gold_spans_per_doc) == len(pred_spans_per_doc)

    tp_by_class, fp_by_class, fn_by_class = Counter(), Counter(), Counter()
    for gold, pred in zip(gold_spans_per_doc, pred_spans_per_doc):
        gold_list, pred_list = list(gold), list(pred)
        candidates = []  # (overlap_len, gold_idx, pred_idx)
        for gi, (gs, ge, gl) in enumerate(gold_list):
            for pi, (ps, pe, pl) in enumerate(pred_list):
                if gl != pl:
                    continue
                if gs < pe and ps < ge:  # half-open interval overlap test
                    candidates.append((min(ge, pe) - max(gs, ps), gi, pi))
        candidates.sort(key=lambda c: -c[0])

        matched_gold, matched_pred = set(), set()
        for _, gi, pi in candidates:
            if gi in matched_gold or pi in matched_pred:
                continue
            matched_gold.add(gi)
            matched_pred.add(pi)
            tp_by_class[gold_list[gi][2]] += 1

        for gi, (_, _, gl) in enumerate(gold_list):
            if gi not in matched_gold:
                fn_by_class[gl] += 1
        for pi, (_, _, pl) in enumerate(pred_list):
            if pi not in matched_pred:
                fp_by_class[pl] += 1

    return _aggregate(tp_by_class, fp_by_class, fn_by_class)
