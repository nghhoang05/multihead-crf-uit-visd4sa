"""
Smoke test for the CRF multi-head span detection model (src/multihead_model.py,
src/multihead_dataset.py, src/multihead_training.py). No GPU/network
required (use_contextual=False throughout).
Run: python scripts/smoke_test_multihead_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.multihead_dataset import MultiHeadCollator, MultiHeadSpanDataset
from src.multihead_model import BiLSTMMultiHeadCRFTagger
from src.multihead_training import _build_multihead_param_groups, merge_aspect_polarity_spans, predict_dataset, train_model
from src.span_dataset import Vocab, load_vocab

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and status == FAIL else ""))


# ---------------------------------------------------------------------------
# 1. merge_aspect_polarity_spans
# ---------------------------------------------------------------------------
aspect_spans = [{"start": 0, "end": 10, "label": "BATTERY"}]
polarity_spans_full_overlap = [{"start": 0, "end": 10, "label": "POSITIVE"}]
check(
    "aspect span with an exactly-matching polarity span merges into ASPECT#POLARITY",
    merge_aspect_polarity_spans(aspect_spans, polarity_spans_full_overlap) == {(0, 10, "BATTERY#POSITIVE")},
)

polarity_spans_partial = [{"start": 8, "end": 20, "label": "NEUTRAL"}, {"start": 0, "end": 6, "label": "NEGATIVE"}]
# overlap with [0,6): min(10,6)-max(0,0)=6 ; overlap with [8,20): min(10,20)-max(0,8)=2 -> NEGATIVE wins (bigger overlap)
check(
    "when multiple polarity spans overlap, the one with the LARGEST overlap wins",
    merge_aspect_polarity_spans(aspect_spans, polarity_spans_partial) == {(0, 10, "BATTERY#NEGATIVE")},
)

check(
    "no overlapping polarity span at all falls back to default_polarity (never drops the aspect span)",
    merge_aspect_polarity_spans(aspect_spans, [{"start": 50, "end": 60, "label": "NEUTRAL"}], default_polarity="POSITIVE")
    == {(0, 10, "BATTERY#POSITIVE")},
)

check(
    "multiple aspect spans are each merged independently",
    merge_aspect_polarity_spans(
        [{"start": 0, "end": 5, "label": "CAMERA"}, {"start": 10, "end": 15, "label": "SCREEN"}],
        [{"start": 0, "end": 5, "label": "NEGATIVE"}, {"start": 10, "end": 15, "label": "POSITIVE"}],
    ) == {(0, 5, "CAMERA#NEGATIVE"), (10, 15, "SCREEN#POSITIVE")},
)

check("merge_aspect_polarity_spans on empty aspect spans returns an empty set", merge_aspect_polarity_spans([], []) == set())

# ---------------------------------------------------------------------------
# 2. BiLSTMMultiHeadCRFTagger end-to-end (loss + backward + predict), use_contextual=False
# ---------------------------------------------------------------------------
syllable_vocab = Vocab(["<PAD>", "<UNK>", "máy", "pin", "hoà"])
char_vocab = Vocab(["<PAD>", "<UNK>"] + list("máypinhoà"))
NUM_TAGS_ASPECT, NUM_TAGS_POLARITY = 22, 8  # PAD + ("O" + 10*(B,I)) = 22 ; PAD + ("O" + 3*(B,I)) = 8

mh_model = BiLSTMMultiHeadCRFTagger(
    syllable_vocab_size=len(syllable_vocab), char_vocab_size=len(char_vocab),
    num_tags_aspect=NUM_TAGS_ASPECT, num_tags_polarity=NUM_TAGS_POLARITY,
    use_char=True, use_contextual=False, lstm_hidden=8,
)
mh_batch = {
    "syllable_ids": torch.tensor([[2, 3, 0]]),
    "char_ids": torch.zeros(1, 3, 4, dtype=torch.long),
    "char_lengths": torch.ones(1, 3, dtype=torch.long),
    "mask": torch.tensor([[True, True, False]]),
    "tag_ids_aspect": torch.tensor([[2, 3, 0]]),   # e.g. B-BATTERY, I-BATTERY, PAD
    "tag_ids_polarity": torch.tensor([[2, 3, 0]]),  # e.g. B-NEGATIVE, I-NEGATIVE, PAD
}
mh_loss = mh_model.loss(mh_batch)
check("BiLSTMMultiHeadCRFTagger.loss() runs and returns a finite scalar", torch.isfinite(mh_loss).item(), str(mh_loss.item()))
mh_loss.backward()
check(
    "gradient flows into the shared encoder (embedding_fusion) from the COMBINED 2-head loss",
    mh_model.embedding_fusion.char_encoder.embedding.weight.grad is not None,
)
check("gradient flows into the aspect CRF head's projection", mh_model.hidden2tag_aspect.weight.grad is not None)
check("gradient flows into the polarity CRF head's projection", mh_model.hidden2tag_polarity.weight.grad is not None)
check("gradient flows into the aspect CRF's own transition parameters", mh_model.crf_aspect.transitions.grad is not None)
check("gradient flows into the polarity CRF's own transition parameters", mh_model.crf_polarity.transitions.grad is not None)

aspect_paths, polarity_paths = mh_model.predict(mh_batch)
check("predict() returns 2 lists (aspect_paths, polarity_paths), one entry per batch example", len(aspect_paths) == 1 and len(polarity_paths) == 1)
check("each decoded path is truncated to the real (unpadded) length (2, not 3)", len(aspect_paths[0]) == 2 and len(polarity_paths[0]) == 2, str((aspect_paths[0], polarity_paths[0])))

# ---------------------------------------------------------------------------
# 3. MultiHeadSpanDataset + MultiHeadCollator on REAL data (UIT-ViSD4SA/iob/*.json)
# ---------------------------------------------------------------------------
real_iob_train = Path("UIT-ViSD4SA/iob/train.json")
real_vocab_path = Path("UIT-ViSD4SA/iob/vocab.json")
if real_iob_train.exists() and real_vocab_path.exists():
    real_vocab = load_vocab(real_vocab_path)
    check(
        "the real vocab.json exposes all 3 tag schemes (aspect, polarity, aspect_polarity)",
        set(real_vocab["tag"]) == {"aspect", "polarity", "aspect_polarity"},
    )
    check(
        "aspect tag vocab has 22 entries (PAD + O + 10 aspects x B/I)",
        len(real_vocab["tag"]["aspect"]) == 22, str(len(real_vocab["tag"]["aspect"])),
    )
    check(
        "polarity tag vocab has 8 entries (PAD + O + 3 polarities x B/I)",
        len(real_vocab["tag"]["polarity"]) == 8, str(len(real_vocab["tag"]["polarity"])),
    )

    real_ds = MultiHeadSpanDataset(real_iob_train)
    check("MultiHeadSpanDataset loads the expected number of real training docs (7785)", len(real_ds) == 7785, str(len(real_ds)))

    real_collator = MultiHeadCollator(real_vocab, tokenizer=None, use_contextual=False)
    real_batch = real_collator([real_ds[0], real_ds[1], real_ds[2]])
    check(
        "collated real batch has all 3 tag tensors with consistent batch/seq shape",
        real_batch["tag_ids_aspect"].shape == real_batch["tag_ids_polarity"].shape == real_batch["tag_ids_combined"].shape
        == real_batch["syllable_ids"].shape,
    )

    real_model = BiLSTMMultiHeadCRFTagger(
        syllable_vocab_size=len(real_vocab["syllable"]), char_vocab_size=len(real_vocab["char"]),
        num_tags_aspect=len(real_vocab["tag"]["aspect"]), num_tags_polarity=len(real_vocab["tag"]["polarity"]),
        use_char=True, use_contextual=False, lstm_hidden=8,
    )
    real_loss = real_model.loss(real_batch)
    check("loss() runs on a real collated batch (3 real docs) without error", torch.isfinite(real_loss).item(), str(real_loss.item()))

    from torch.utils.data import DataLoader
    real_loader = DataLoader([real_ds[0], real_ds[1], real_ds[2]], batch_size=3, collate_fn=real_collator)
    gold_real, pred_real, doc_ids_real = predict_dataset(real_model, real_loader, real_vocab, torch.device("cpu"), use_amp=False, log_fn=lambda *_a: None)
    check(
        "predict_dataset returns one gold/pred entry per real doc, each a set of (start,end,label) tuples",
        len(gold_real) == 3 and len(pred_real) == 3 and all(isinstance(s, set) for s in gold_real + pred_real),
    )
    check(
        "every gold span decoded from tag_ids_combined has the ASPECT#POLARITY format (contains '#')",
        all("#" in label for doc_spans in gold_real for (_, _, label) in doc_spans),
    )
else:
    check("UIT-ViSD4SA/iob/{train,vocab}.json not found -- skipped real-data test (not a failure, just unavailable here)", True)

# ---------------------------------------------------------------------------
# 4. src/multihead_training.py -- end-to-end train_model (CPU, tiny synthetic data)
# ---------------------------------------------------------------------------
from torch.utils.data import DataLoader

toy_docs = [
    {"doc_id": "d0", "text": "máy pin", "tokens": ["máy", "pin"], "token_offsets": [(0, 3), (4, 7)],
     "tags_aspect": ["B-BATTERY", "I-BATTERY"], "tags_polarity": ["B-POSITIVE", "I-POSITIVE"],
     "tags_aspect_polarity": ["B-BATTERY#POSITIVE", "I-BATTERY#POSITIVE"]},
    {"doc_id": "d1", "text": "hoà bình thật", "tokens": ["hoà", "bình", "thật"],
     "token_offsets": [(0, 3), (4, 8), (9, 13)],
     "tags_aspect": ["O", "O", "O"], "tags_polarity": ["O", "O", "O"], "tags_aspect_polarity": ["O", "O", "O"]},
]
toy_vocab = {
    "syllable": syllable_vocab, "char": char_vocab,
    "tag": {
        "aspect": Vocab(["<PAD>", "O", "B-BATTERY", "I-BATTERY"]),
        "polarity": Vocab(["<PAD>", "O", "B-POSITIVE", "I-POSITIVE"]),
        "aspect_polarity": Vocab(["<PAD>", "O", "B-BATTERY#POSITIVE", "I-BATTERY#POSITIVE"]),
    },
}
toy_collator = MultiHeadCollator(toy_vocab, tokenizer=None, use_contextual=False)
toy_loader = DataLoader(toy_docs, batch_size=2, collate_fn=toy_collator)

toy_model = BiLSTMMultiHeadCRFTagger(
    syllable_vocab_size=len(toy_vocab["syllable"]), char_vocab_size=len(toy_vocab["char"]),
    num_tags_aspect=len(toy_vocab["tag"]["aspect"]), num_tags_polarity=len(toy_vocab["tag"]["polarity"]),
    use_char=True, use_contextual=False, lstm_hidden=8,
)
quiet_log_mh = []
mh_train_result = train_model(
    toy_model, toy_loader, toy_loader, toy_vocab, torch.device("cpu"),
    epochs=1, checkpoint_path=None, log_fn=quiet_log_mh.append,
)
check(
    "multihead_training.train_model runs 1 epoch end-to-end on CPU without error",
    "history" in mh_train_result and len(mh_train_result["history"]) == 1,
    str(mh_train_result),
)
check("train_model logs mixed precision as OFF on CPU (correct auto-detection)", any("off" in line for line in quiet_log_mh))

pg_mh = _build_multihead_param_groups(toy_model, lr=1e-3, xlmr_lr=2e-5, weight_decay=1e-2)
check(
    "_build_multihead_param_groups gives 2 groups when use_contextual=False (non-CRF + the 2 CRFs' own params)",
    len(pg_mh) == 2 and pg_mh[1]["weight_decay"] == 1e-2,
    str(pg_mh),
)

# ---------------------------------------------------------------------------
n_fail = sum(1 for _, status, _ in results if status == FAIL)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
sys.exit(1 if n_fail else 0)
