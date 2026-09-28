"""
CLI to train the proposed model (`BiLSTMMultiHeadCRFTagger`, 2 independent
CRF heads) -- reads all hyperparameters from `config/hyperparams.yaml` and
calls `src/multihead_training.py` directly (no duplicated training/model logic).

Usage:
    python scripts/train.py --seeds 42 123 777 2024 2025

Requires `UIT-ViSD4SA/iob/{train,dev,test,vocab}.json` (run
`scripts/prepare_data.py` first) and PhoW2V downloaded + extracted
(`--phow2v-dir`, see README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.pretrained_syllable_embedding import build_syllable_embedding_matrix, load_word2vec_vectors
from src.span_dataset import load_vocab
from src.token_batch_sampler import TokenBudgetBatchSampler
from src.training import set_seed


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_multihead_loaders(vocab, iob_dir: Path, tokenizer, token_budget: int, use_contextual: bool, max_train_docs: int | None = None):
    from src.multihead_dataset import MultiHeadCollator, MultiHeadSpanDataset

    train_ds = MultiHeadSpanDataset(iob_dir / "train.json")
    dev_ds = MultiHeadSpanDataset(iob_dir / "dev.json")
    test_ds = MultiHeadSpanDataset(iob_dir / "test.json")
    collate = MultiHeadCollator(vocab, tokenizer=tokenizer, use_contextual=use_contextual)

    if max_train_docs is not None:
        # Truncate BEFORE computing train_lengths/building the sampler -- truncating
        # after would leave TokenBudgetBatchSampler yielding out-of-range indices.
        train_ds.docs = train_ds.docs[:max_train_docs]

    train_lengths = [len(d["tokens"]) for d in train_ds.docs]
    dev_lengths = [len(d["tokens"]) for d in dev_ds.docs]
    test_lengths = [len(d["tokens"]) for d in test_ds.docs]

    train_loader = DataLoader(train_ds, collate_fn=collate,
                               batch_sampler=TokenBudgetBatchSampler(train_lengths, token_budget=token_budget, shuffle=True, seed=42))
    dev_loader = DataLoader(dev_ds, collate_fn=collate,
                             batch_sampler=TokenBudgetBatchSampler(dev_lengths, token_budget=token_budget, shuffle=False))
    test_loader = DataLoader(test_ds, collate_fn=collate,
                              batch_sampler=TokenBudgetBatchSampler(test_lengths, token_budget=token_budget, shuffle=False))
    return train_ds, dev_ds, test_ds, train_loader, dev_loader, test_loader


def run_one_seed_multihead(seed, vocab, syllable_matrix, arch, opt, train_cfg, loaders, device, output_dir, use_amp):
    """`loaders`: full 6-tuple from `build_multihead_loaders` (train_ds, dev_ds, test_ds, train_loader, dev_loader, test_loader)."""
    from src.evaluation import evaluate, save_raw_predictions
    from src.multihead_model import BiLSTMMultiHeadCRFTagger
    from src.multihead_training import aggregate_merge_stats, predict_dataset, train_model

    train_ds, dev_ds, test_ds, train_loader, dev_loader, test_loader = loaders
    set_seed(seed)

    model = BiLSTMMultiHeadCRFTagger(
        syllable_vocab_size=len(vocab["syllable"]), char_vocab_size=len(vocab["char"]),
        num_tags_aspect=len(vocab["tag"]["aspect"]), num_tags_polarity=len(vocab["tag"]["polarity"]),
        use_char=arch["use_char"], use_contextual=arch["use_contextual"],
        contextual_model_name=arch["contextual_model_name"], contextual_projected_dim=arch["contextual_projected_dim"],
        lstm_hidden=arch["lstm_hidden"], dropout=arch["dropout"],
        pretrained_syllable_matrix=syllable_matrix,
        freeze_syllable=arch["freeze_syllable"], freeze_contextual=arch["freeze_contextual"],
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    checkpoint_path = output_dir / f"checkpoint_multihead_seed{seed}.pt"
    t0 = time.time()
    result = train_model(
        model, train_loader, dev_loader, vocab, device,
        epochs=train_cfg["epochs"], lr=opt["lr"], xlmr_lr=opt["xlmr_lr"], weight_decay=opt["weight_decay"],
        grad_clip=opt["grad_clip"], checkpoint_path=checkpoint_path, use_amp=use_amp,
    )
    train_time_min = (time.time() - t0) / 60

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gold_test, pred_test, _, test_details = predict_dataset(model, test_loader, vocab, device, collect_details=True)
    test_metrics = evaluate(gold_test, pred_test)
    merge_stats_seed = aggregate_merge_stats([d["merge_stats"] for d in test_details])

    predictions_path = output_dir / f"test_predictions_multihead_seed{seed}.jsonl"
    save_raw_predictions(test_details, predictions_path)

    results_payload = {
        "config": {
            "model": "BiLSTM + 2-CRF-head (aspect + polarity)", "scheme": "aspect_polarity", "seed": seed,
            "epochs_ran": len(result["history"]), "epochs_max": train_cfg["epochs"],
            "n_total_params": n_total, "n_trainable_params": n_trainable,
        },
        "history": result["history"],
        "best_dev_macro_f1": result["best_dev_macro_f1"],
        "test_micro": test_metrics["micro"], "test_macro": test_metrics["macro"], "test_per_class": test_metrics["per_class"],
        "merge_stats": merge_stats_seed,
        "train_time_min": train_time_min,
    }
    results_path = output_dir / f"results_multihead_seed{seed}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results_payload, results_path, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/hyperparams.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("UIT-ViSD4SA/iob"))
    parser.add_argument("--phow2v-dir", type=Path, default=Path("phow2v/extracted"),
                         help="Extracted PhoW2V directory (ignored with --no-pretrained-syllable)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: seeds from config/hyperparams.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count (default: from config)")
    parser.add_argument("--no-contextual", action="store_true", help="Disable XLM-R -- for quick CPU testing ONLY, do NOT use to reproduce paper results")
    parser.add_argument("--no-pretrained-syllable", action="store_true", help="Skip PhoW2V (random-init) -- for quick testing ONLY")
    parser.add_argument("--lstm-hidden", type=int, default=None, help="Override lstm_hidden -- for quick testing ONLY")
    parser.add_argument("--max-train-docs", type=int, default=None, help="Cap the number of training documents (quick testing, do NOT use to reproduce results)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    arch = dict(cfg["architecture"])
    opt = cfg["optimizer"]
    train_cfg = dict(cfg["training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.no_contextual:
        arch["use_contextual"] = False
    if args.lstm_hidden is not None:
        arch["lstm_hidden"] = args.lstm_hidden
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seeds: {seeds}")

    vocab = load_vocab(args.data_dir / "vocab.json")
    print(f"Syllable vocab: {len(vocab['syllable']):,} | Char vocab: {len(vocab['char']):,}")

    syllable_matrix = None
    if not args.no_pretrained_syllable:
        kv = load_word2vec_vectors(args.phow2v_dir)
        syllable_matrix, coverage_stats = build_syllable_embedding_matrix(vocab["syllable"], kv)
        print(f"PhoW2V coverage: {coverage_stats['coverage']:.1%}")

    tokenizer = AutoTokenizer.from_pretrained(arch["contextual_model_name"]) if arch["use_contextual"] else None

    if args.max_train_docs is not None:
        print(f"!! --max-train-docs is ON: using only {args.max_train_docs} training documents -- do NOT report this result.")

    loaders = build_multihead_loaders(vocab, args.data_dir, tokenizer, cfg["data"]["token_budget"], arch["use_contextual"], args.max_train_docs)
    print(f"train={len(loaders[0]):,}  dev={len(loaders[1]):,}  test={len(loaders[2]):,}")

    for seed in seeds:
        print(f"\n{'#' * 70}\nSEED = {seed}\n{'#' * 70}")
        results_payload, results_path, checkpoint_path = run_one_seed_multihead(
            seed, vocab, syllable_matrix, arch, opt, train_cfg, loaders, device, args.output_dir, train_cfg["use_amp"],
        )
        print(f"TEST (seed={seed}) -> micro F1={results_payload['test_micro']['f1']:.4f}  "
              f"macro F1={results_payload['test_macro']['f1']:.4f}")
        print(f"Saved: {results_path} | {checkpoint_path}")


if __name__ == "__main__":
    main()
