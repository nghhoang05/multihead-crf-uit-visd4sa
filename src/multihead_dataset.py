"""
Dataset/collate utilities for the CRF multi-head span detection variant
(`src/multihead_model.py::BiLSTMMultiHeadCRFTagger`) -- configuration #2 in
this project's 3-way span-detection ablation (see that module's docstring).

Reuses the EXACT SAME `UIT-ViSD4SA/iob/{train,dev,test}.json` + `vocab.json`
files as the baseline (`src/span_dataset.py`), no new preprocessing needed:
every doc there already carries `tags_aspect`, `tags_polarity`, AND
`tags_aspect_polarity` (the 3 schemes from `src/span_detection.py`), and
`vocab.json`'s `tag_vocab` already has all 3 as separate tag sets. This
collator reads all three per example -- `tags_aspect`/`tags_polarity` are
the two heads' TRAINING targets, while `tags_aspect_polarity` is carried
through unchanged so gold spans can be decoded EXACTLY like the baseline
does (same tag vocab, same `bio_to_spans`), keeping the 3-way ablation's
gold side identical across all configurations.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.span_dataset import _build_alignment_matrix, _encode_syllable_char_mask


class MultiHeadSpanDataset(Dataset):
    def __init__(self, iob_path: str | Path):
        with open(iob_path, encoding="utf-8") as f:
            data = json.load(f)
        self.docs = data["docs"]

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        return {
            "doc_id": doc["doc_id"], "text": doc["text"],
            "tokens": doc["tokens"], "token_offsets": doc["token_offsets"],
            "tags_aspect": doc["tags_aspect"], "tags_polarity": doc["tags_polarity"],
            "tags_aspect_polarity": doc["tags_aspect_polarity"],
        }


class MultiHeadCollator:
    """Callable collate_fn, sharing its syllable/char/mask encoding with
    `src/span_dataset.py::Collator` via `_encode_syllable_char_mask` (and its
    XLM-R subword alignment matrix construction via `_build_alignment_matrix`)
    -- only the TAG encoding differs (3 tag tensors instead of 1)."""

    def __init__(self, vocab: dict, tokenizer=None, use_contextual: bool = True, max_subword_len: int = 512):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.use_contextual = use_contextual
        self.max_subword_len = max_subword_len

    def __call__(self, batch: list[dict]) -> dict:
        bsz = len(batch)
        seq_len = max(len(ex["tokens"]) for ex in batch)
        max_char_len = max((len(tok) for ex in batch for tok in ex["tokens"]), default=1)

        syll_vocab, char_vocab = self.vocab["syllable"], self.vocab["char"]
        aspect_tag_vocab = self.vocab["tag"]["aspect"]
        polarity_tag_vocab = self.vocab["tag"]["polarity"]
        combined_tag_vocab = self.vocab["tag"]["aspect_polarity"]

        syllable_ids, mask, char_ids, char_lengths = _encode_syllable_char_mask(
            batch, syll_vocab, char_vocab, bsz, seq_len, max_char_len,
        )
        tag_ids_aspect = torch.zeros(bsz, seq_len, dtype=torch.long)
        tag_ids_polarity = torch.zeros(bsz, seq_len, dtype=torch.long)
        tag_ids_combined = torch.zeros(bsz, seq_len, dtype=torch.long)
        for b, ex in enumerate(batch):
            for i in range(len(ex["tokens"])):
                tag_ids_aspect[b, i] = aspect_tag_vocab.encode(ex["tags_aspect"][i])
                tag_ids_polarity[b, i] = polarity_tag_vocab.encode(ex["tags_polarity"][i])
                tag_ids_combined[b, i] = combined_tag_vocab.encode(ex["tags_aspect_polarity"][i])

        result = {
            "syllable_ids": syllable_ids, "mask": mask,
            "tag_ids_aspect": tag_ids_aspect, "tag_ids_polarity": tag_ids_polarity,
            "tag_ids_combined": tag_ids_combined,
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
