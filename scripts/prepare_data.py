"""
Converts UIT-ViSD4SA (original jsonl format) into the syllable-level IOB +
vocab that `src/multihead_dataset.py`/`src/span_dataset.py` need.
`train.json`/`dev.json`/`test.json` are not committed to this repo (third-
party research data, citation required) -- download the raw data yourself
and run this script.

Usage:
    1. Download `train.jsonl`/`dev.jsonl`/`test.jsonl` from
       https://github.com/kimkim00/UIT-ViSD4SA.
    2. Put all 3 files in one directory, e.g. `raw_data/`.
    3. Run:
         python scripts/prepare_data.py --raw-dir raw_data --out-dir UIT-ViSD4SA/iob

Each raw jsonl line: {"text": "...", "labels": [[start, end, "ASPECT#POLARITY"], ...]}
(no "doc_id" -- this script generates one as "<split>_<i>").
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.span_detection import convert_dataset

SPLITS = ("train", "dev", "test")
SCHEMES = ("aspect", "polarity", "aspect_polarity")
SPECIAL_TOKENS = ["<PAD>", "<UNK>"]


def load_raw_split(raw_dir: Path, split: str) -> list[dict]:
    path = raw_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- download the raw data from "
            f"https://github.com/kimkim00/UIT-ViSD4SA and place it in {raw_dir} first."
        )
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            docs.append({"doc_id": f"{split}_{i}", "text": raw["text"], "labels": raw["labels"]})
    return docs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw train.jsonl/dev.jsonl/test.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("UIT-ViSD4SA/iob"), help="Output directory (default UIT-ViSD4SA/iob)")
    args = parser.parse_args()

    raw = {split: load_raw_split(args.raw_dir, split) for split in SPLITS}
    for split in SPLITS:
        print(f"{split}: {len(raw[split]):,} reviews")

    print("\nTokenizing into syllables + converting to IOB for all 3 label schemes...")
    tagged = {split: {scheme: convert_dataset(raw[split], scheme=scheme) for scheme in SCHEMES} for split in SPLITS}

    print("Building vocabulary (syllable + char) from the train split...")
    syllable_counter, char_counter = Counter(), Counter()
    for t in tagged["train"]["aspect_polarity"]:
        syllable_counter.update(tok.lower() for tok in t.tokens)
        for tok in t.tokens:
            char_counter.update(tok)
    syllable_vocab = SPECIAL_TOKENS + [w for w, _ in syllable_counter.most_common()]
    char_vocab = SPECIAL_TOKENS + [c for c, _ in char_counter.most_common()]
    print(f"  Syllable vocab: {len(syllable_vocab):,} (including PAD/UNK)")
    print(f"  Char vocab: {len(char_vocab):,} (including PAD/UNK)")

    tag_vocab = {}
    for scheme in SCHEMES:
        tags = set()
        for split in SPLITS:
            for t in tagged[split][scheme]:
                tags.update(t.tags)
        tag_vocab[scheme] = ["O"] + sorted(tags - {"O"})
        print(f"  tag_vocab[{scheme}]: {len(tag_vocab[scheme])} tags")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        docs_out = []
        for i, doc in enumerate(raw[split]):
            entry = {
                "doc_id": doc["doc_id"], "text": doc["text"],
                "tokens": tagged[split]["aspect_polarity"][i].tokens,
                "token_offsets": tagged[split]["aspect_polarity"][i].token_offsets,
            }
            for scheme in SCHEMES:
                entry[f"tags_{scheme}"] = tagged[split][scheme][i].tags
            docs_out.append(entry)
        out_path = args.out_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"split": split, "n_docs": len(docs_out), "docs": docs_out}, f, ensure_ascii=False, indent=2)
        print(f"Saved {out_path}: {len(docs_out):,} documents")

    vocab_path = args.out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"syllable_vocab": syllable_vocab, "char_vocab": char_vocab, "tag_vocab": tag_vocab}, f, ensure_ascii=False, indent=2)
    print(f"Saved {vocab_path}")
    print(
        "\nNote: this conversion does NOT apply the original project's '122 offset-bug fix' step "
        "-- if you need to EXACTLY match the reported results, use the original train/dev/test.json "
        "files instead of regenerating them with this script."
    )


if __name__ == "__main__":
    main()
