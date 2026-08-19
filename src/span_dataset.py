"""
PyTorch Dataset/collate utilities that turn the syllable-tokenized, IOB-tagged
documents produced by `src/span_detection.py` (see notebook 08) into padded
batches consumable by `src/bilstm_crf.py`'s `BiLSTMCRFTagger`.

Handles the one genuinely tricky bit: aligning XLM-R SUBWORD embeddings back
to SYLLABLE tokens (the paper's tagging granularity) by mean-pooling every
subword whose character span overlaps a given syllable's character span.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, UNK = "<PAD>", "<UNK>"


class Vocab:
    """Simple word<->id lookup with index 0 reserved for PAD. Syllable/char
    vocabs additionally reserve index 1 for UNK (out-of-vocabulary tokens
    are expected at inference time); tag vocabs do not need UNK since the
    tag set is closed and fully known ahead of time."""

    def __init__(self, tokens: list[str]):
        assert tokens[0] == PAD, "vocab must start with <PAD> at index 0"
        self.itos = tokens
        self.stoi = {tok: i for i, tok in enumerate(tokens)}

    def __len__(self):
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi.get(UNK, 0))

    def decode(self, idx: int) -> str:
        return self.itos[idx]


def load_vocab(vocab_path: Path) -> dict:
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "syllable": Vocab(raw["syllable_vocab"]),
        "char": Vocab(raw["char_vocab"]),
        "tag": {scheme: _tag_vocab(tags) for scheme, tags in raw["tag_vocab"].items()},
    }


def _tag_vocab(tags: list[str]) -> Vocab:
    """Tag vocabs don't have PAD/UNK in vocab.json (they store ['O', 'B-...', ...]);
    wrap with a PAD entry at index 0 for batching (padded positions are masked
    out of the loss/decoding anyway, PAD is never a real prediction target)."""
    return Vocab([PAD, "O"] + [t for t in tags if t != "O"])


class SpanDataset(Dataset):
    def __init__(self, iob_path: Path, scheme: str = "aspect_polarity"):
        with open(iob_path, encoding="utf-8") as f:
            data = json.load(f)
        self.docs = data["docs"]
        self.scheme = scheme

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        return {
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "tokens": doc["tokens"],
            "token_offsets": doc["token_offsets"],
            "tags": doc[f"tags_{self.scheme}"],
        }


def _encode_syllable_char_mask(
    batch: list[dict], syll_vocab: "Vocab", char_vocab: "Vocab", bsz: int, seq_len: int, max_char_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared by `Collator` (this file) and `src/multihead_dataset.py::
    MultiHeadCollator` -- the syllable-id/char-id/mask encoding is IDENTICAL
    between the two collators (only how TAGS are encoded differs: 1 tensor
    vs 3), so this factors out the common loop instead of keeping 2
    independently-maintained copies of it (previously `MultiHeadCollator`
    had its own verbatim copy of this loop -- a real risk if this logic were
    ever fixed in one place and not the other). Returns (syllable_ids, mask,
    char_ids, char_lengths), same tensors/shapes/semantics as before this
    refactor (lowercased syllable lookup, per-character char_vocab encoding,
    `mask[b, :n] = True` for the real, unpadded length of each example)."""
    syllable_ids = torch.zeros(bsz, seq_len, dtype=torch.long)
    mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
    char_ids = torch.zeros(bsz, seq_len, max_char_len, dtype=torch.long)
    char_lengths = torch.ones(bsz, seq_len, dtype=torch.long)

    for b, ex in enumerate(batch):
        n = len(ex["tokens"])
        mask[b, :n] = True
        for i, tok in enumerate(ex["tokens"]):
            syllable_ids[b, i] = syll_vocab.encode(tok.lower())
            char_lengths[b, i] = max(len(tok), 1)
            for c, ch in enumerate(tok):
                char_ids[b, i, c] = char_vocab.encode(ch)
    return syllable_ids, mask, char_ids, char_lengths


def _build_alignment_matrix(token_offsets, subword_offsets, seq_len, subword_len):
    """(seq_len, subword_len) mean-pooling matrix: row i has 1/k at each of the
    k subwords whose char span overlaps token i's char span; all-zero row if
    no subword overlaps (e.g. token was truncated away by max_length)."""
    matrix = torch.zeros(seq_len, subword_len)
    for i, (tok_start, tok_end) in enumerate(token_offsets):
        overlaps = [
            j for j, (sw_start, sw_end) in enumerate(subword_offsets)
            if sw_end > sw_start and sw_end > tok_start and sw_start < tok_end
        ]
        if overlaps:
            matrix[i, overlaps] = 1.0 / len(overlaps)
    return matrix


class Collator:
    """Callable collate_fn: needs the vocabs (built in notebook 08) and,
    when `use_contextual=True`, an XLM-R tokenizer (fast, for offset_mapping)
    to build the syllable<->subword alignment matrix per example."""

    def __init__(self, vocab: dict, scheme: str, tokenizer=None, use_contextual: bool = True, max_subword_len: int = 512):
        self.vocab = vocab
        self.scheme = scheme
        self.tokenizer = tokenizer
        self.use_contextual = use_contextual
        self.max_subword_len = max_subword_len

    def __call__(self, batch: list[dict]) -> dict:
        bsz = len(batch)
        seq_len = max(len(ex["tokens"]) for ex in batch)
        max_char_len = max((len(tok) for ex in batch for tok in ex["tokens"]), default=1)

        syll_vocab, char_vocab = self.vocab["syllable"], self.vocab["char"]
        tag_vocab = self.vocab["tag"][self.scheme]

        syllable_ids, mask, char_ids, char_lengths = _encode_syllable_char_mask(
            batch, syll_vocab, char_vocab, bsz, seq_len, max_char_len,
        )
        tag_ids = torch.zeros(bsz, seq_len, dtype=torch.long)
        for b, ex in enumerate(batch):
            for i in range(len(ex["tokens"])):
                tag_ids[b, i] = tag_vocab.encode(ex["tags"][i])

        result = {
            "syllable_ids": syllable_ids, "tag_ids": tag_ids, "mask": mask,
            "char_ids": char_ids, "char_lengths": char_lengths,
            "doc_ids": [ex["doc_id"] for ex in batch], "texts": [ex["text"] for ex in batch],
            "tokens": [ex["tokens"] for ex in batch], "token_offsets": [ex["token_offsets"] for ex in batch],
        }

        if self.use_contextual:
            enc = self.tokenizer(
                [ex["text"] for ex in batch], return_offsets_mapping=True, return_tensors="pt",
                padding=True, truncation=True, max_length=self.max_subword_len,
            )
            subword_len = enc["input_ids"].shape[1]
            alignment = torch.zeros(bsz, seq_len, subword_len)
            for b, ex in enumerate(batch):
                sw_offsets = enc["offset_mapping"][b].tolist()
                alignment[b] = _build_alignment_matrix(ex["token_offsets"], sw_offsets, seq_len, subword_len)
            result["subword_ids"] = enc["input_ids"]
            result["subword_mask"] = enc["attention_mask"]
            result["alignment"] = alignment

        return result
