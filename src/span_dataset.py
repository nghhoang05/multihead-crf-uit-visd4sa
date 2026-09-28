"""
Vocab loading + the syllable/char/subword-alignment encoding shared by
`src/multihead_dataset.py::MultiHeadCollator` -- turns the syllable-
tokenized, IOB-tagged documents produced by `src/span_detection.py` into
padded batches. Aligns XLM-R subword embeddings back to syllable tokens by
mean-pooling every subword whose character span overlaps a syllable's.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

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


def _encode_syllable_char_mask(
    batch: list[dict], syll_vocab: "Vocab", char_vocab: "Vocab", bsz: int, seq_len: int, max_char_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Used by `src/multihead_dataset.py::MultiHeadCollator`. Returns
    (syllable_ids, mask, char_ids, char_lengths)."""
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
