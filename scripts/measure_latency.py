"""
Measures inference latency of the proposed model (Section IV-B), 2
independent CRF heads with the `merge_aspect_polarity_spans` step --
**batch_size=1** (real per-sentence latency, not the dynamic token-budget
batching used for training/F1 evaluation). Prints a per-stage breakdown
(model forward, span decode, merge) to show the merge step's own cost.

Usage:
    python scripts/measure_latency.py --checkpoint checkpoint_multihead_seed42.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.inference_latency import measure_latency_multihead, print_latency_report
from src.multihead_dataset import MultiHeadCollator, MultiHeadSpanDataset
from src.multihead_model import BiLSTMMultiHeadCRFTagger
from src.span_dataset import load_vocab


def _build_multihead(vocab, arch, checkpoint_path, data_dir, tokenizer, device, n_docs):
    test_ds = MultiHeadSpanDataset(data_dir / "test.json")
    if n_docs is not None:
        test_ds.docs = test_ds.docs[:n_docs]
    collate = MultiHeadCollator(vocab, tokenizer=tokenizer, use_contextual=arch["use_contextual"])
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate)  # batch_size=1 -- REQUIRED for this measurement

    model = BiLSTMMultiHeadCRFTagger(
        syllable_vocab_size=len(vocab["syllable"]), char_vocab_size=len(vocab["char"]),
        num_tags_aspect=len(vocab["tag"]["aspect"]), num_tags_polarity=len(vocab["tag"]["polarity"]),
        use_char=arch["use_char"], use_contextual=arch["use_contextual"],
        contextual_model_name=arch["contextual_model_name"], contextual_projected_dim=arch["contextual_projected_dim"],
        lstm_hidden=arch["lstm_hidden"], dropout=arch["dropout"],
        pretrained_syllable_matrix=None,  # checkpoint overwrites everything, no need for real PhoW2V to measure latency
        freeze_syllable=arch["freeze_syllable"], freeze_contextual=arch["freeze_contextual"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/hyperparams.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("UIT-ViSD4SA/iob"))
    parser.add_argument("--n-docs", type=int, default=None, help="Cap the number of test sentences used for timing (default: whole test set)")
    parser.add_argument("--no-contextual", action="store_true",
                         help="Disable XLM-R -- MUST match this checkpoint's training config (only use if it was also trained with --no-contextual)")
    parser.add_argument("--lstm-hidden", type=int, default=None,
                         help="Override lstm_hidden -- MUST match this checkpoint's training config")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    arch = dict(cfg["architecture"])
    if args.no_contextual:
        arch["use_contextual"] = False
    if args.lstm_hidden is not None:
        arch["lstm_hidden"] = args.lstm_hidden
    lat_cfg = cfg["inference_latency"]
    assert lat_cfg["batch_size"] == 1, "This script measures at batch_size=1 per the config only -- see docstring."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.data_dir / "vocab.json")
    tokenizer = AutoTokenizer.from_pretrained(arch["contextual_model_name"]) if arch["use_contextual"] else None

    model, loader = _build_multihead(vocab, arch, args.checkpoint, args.data_dir, tokenizer, device, args.n_docs)
    result = measure_latency_multihead(
        model, loader, vocab["tag"]["aspect"], vocab["tag"]["polarity"], device,
        n_warmup_batches=lat_cfg["n_warmup_batches"], use_amp=lat_cfg["use_amp"],
    )
    print_latency_report(result)

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
