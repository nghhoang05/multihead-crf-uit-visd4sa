"""
Re-evaluates a trained checkpoint (no retraining) on the test set: Exact
Match F1 (micro/macro/per-class, `src/evaluation.py::evaluate`) and the
default-assignment/orphan-polarity rates (Section III-C,
`src/multihead_training.py::aggregate_merge_stats`).

Usage:
    python scripts/evaluate.py --checkpoint checkpoint_multihead_seed42.pt

The model architecture is rebuilt from `config/hyperparams.yaml` (default)
-- it MUST match the architecture used to train that checkpoint, or
`load_state_dict` will raise a shape-mismatch error.
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

from src.evaluation import evaluate
from src.multihead_dataset import MultiHeadCollator, MultiHeadSpanDataset
from src.multihead_model import BiLSTMMultiHeadCRFTagger
from src.multihead_training import aggregate_merge_stats, predict_dataset
from src.pretrained_syllable_embedding import build_syllable_embedding_matrix, load_word2vec_vectors
from src.span_dataset import load_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/hyperparams.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("UIT-ViSD4SA/iob"))
    parser.add_argument("--phow2v-dir", type=Path, default=Path("phow2v/extracted"))
    parser.add_argument("--no-pretrained-syllable", action="store_true",
                         help="Bỏ qua nạp PhoW2V -- AN TOÀN cho eval (checkpoint đã ghi đè toàn bộ trọng số embedding, không cần ma trận pretrained gốc)")
    parser.add_argument("--no-contextual", action="store_true",
                         help="Tắt XLM-R -- PHẢI khớp đúng cấu hình lúc train checkpoint này (chỉ dùng khi checkpoint cũng được train với --no-contextual)")
    parser.add_argument("--lstm-hidden", type=int, default=None,
                         help="Override lstm_hidden -- PHẢI khớp đúng cấu hình lúc train checkpoint này")
    parser.add_argument("--out", type=Path, default=None, help="Lưu kết quả JSON (mặc định: in ra màn hình, không lưu)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    arch = dict(cfg["architecture"])
    if args.no_contextual:
        arch["use_contextual"] = False
    if args.lstm_hidden is not None:
        arch["lstm_hidden"] = args.lstm_hidden

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.data_dir / "vocab.json")

    # The checkpoint already has all learned weights -- no need for real PhoW2V vectors
    # (only the SHAPE matters; random-init is fine since load_state_dict() overwrites it).
    syllable_matrix = None
    if not args.no_pretrained_syllable:
        kv = load_word2vec_vectors(args.phow2v_dir)
        syllable_matrix, _ = build_syllable_embedding_matrix(vocab["syllable"], kv)

    tokenizer = AutoTokenizer.from_pretrained(arch["contextual_model_name"]) if arch["use_contextual"] else None

    test_ds = MultiHeadSpanDataset(args.data_dir / "test.json")
    collate = MultiHeadCollator(vocab, tokenizer=tokenizer, use_contextual=arch["use_contextual"])
    test_loader = DataLoader(test_ds, batch_size=8, collate_fn=collate)

    model = BiLSTMMultiHeadCRFTagger(
        syllable_vocab_size=len(vocab["syllable"]), char_vocab_size=len(vocab["char"]),
        num_tags_aspect=len(vocab["tag"]["aspect"]), num_tags_polarity=len(vocab["tag"]["polarity"]),
        use_char=arch["use_char"], use_contextual=arch["use_contextual"],
        contextual_model_name=arch["contextual_model_name"], contextual_projected_dim=arch["contextual_projected_dim"],
        lstm_hidden=arch["lstm_hidden"], dropout=arch["dropout"],
        pretrained_syllable_matrix=syllable_matrix,
        freeze_syllable=arch["freeze_syllable"], freeze_contextual=arch["freeze_contextual"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    gold, pred, _, details = predict_dataset(model, test_loader, vocab, device, use_amp=False, collect_details=True)
    metrics = evaluate(gold, pred)
    merge_stats = aggregate_merge_stats([d["merge_stats"] for d in details])
    report = {
        "model": "multihead", "checkpoint": str(args.checkpoint),
        "test_micro": metrics["micro"], "test_macro": metrics["macro"], "test_per_class": metrics["per_class"],
        "merge_stats": merge_stats,
    }

    print(f"\n=== {args.checkpoint} ===")
    print(f"Test micro F1: {report['test_micro']['f1']:.4f} | macro F1: {report['test_macro']['f1']:.4f}")
    s = report["merge_stats"]
    print(f"Default-assignment rate: {s['default_assigned_rate_pct']:.2f}% "
          f"({s['default_assigned']:,}/{s['total_aspect_spans']:,} span aspect)")
    print(f"Orphan-polarity rate: {s['orphan_polarity_discarded_rate_pct']:.2f}% "
          f"({s['orphan_polarity_discarded']:,}/{s['total_polarity_spans']:,} span polarity)")

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
