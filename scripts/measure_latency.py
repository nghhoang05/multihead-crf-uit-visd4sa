"""
Đo và so sánh độ trễ suy luận (Section IV-B): baseline (1 CRF, 61 nhãn,
không hợp nhất) vs mô hình đề xuất (2 CRF độc lập 21+7 nhãn, CÓ bước hợp
nhất `merge_aspect_polarity_spans`) -- **batch_size=1** (đo độ trễ THỰC TẾ
1 câu/lần, KHÔNG dùng dynamic token-budget batching của lúc huấn luyện/đánh
giá F1, vốn cho throughput cao hơn nhờ xử lý song song nhiều câu).

Cách dùng:
    python scripts/measure_latency.py \
        --baseline-checkpoint checkpoint_baseline_seed42.pt \
        --multihead-checkpoint checkpoint_multihead_seed42.pt

Chỉ cần 1 trong 2 checkpoint để đo latency CỦA RIÊNG cấu hình đó (bỏ qua
tham số checkpoint còn lại) -- truyền cả 2 để có bảng so sánh đầy đủ
(`print_latency_comparison`), đúng bảng Section IV-B.
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

from src.inference_latency import measure_latency_baseline, measure_latency_multihead, print_latency_comparison
from src.span_dataset import load_vocab


def _build_baseline(vocab, arch, checkpoint_path, data_dir, tokenizer, device, n_docs):
    from src.bilstm_crf import BiLSTMCRFTagger
    from src.span_dataset import Collator, SpanDataset

    scheme = "aspect_polarity"
    test_ds = SpanDataset(data_dir / "test.json", scheme=scheme)
    if n_docs is not None:
        test_ds.docs = test_ds.docs[:n_docs]
    collate = Collator(vocab, scheme=scheme, tokenizer=tokenizer, use_contextual=arch["use_contextual"])
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate)  # batch_size=1 -- YÊU CẦU của phép đo này

    model = BiLSTMCRFTagger(
        syllable_vocab_size=len(vocab["syllable"]), char_vocab_size=len(vocab["char"]),
        num_tags=len(vocab["tag"][scheme]),
        use_char=arch["use_char"], use_contextual=arch["use_contextual"],
        contextual_model_name=arch["contextual_model_name"], contextual_projected_dim=arch["contextual_projected_dim"],
        lstm_hidden=arch["lstm_hidden"], dropout=arch["dropout"],
        pretrained_syllable_matrix=None,  # checkpoint ghi đè toàn bộ, không cần PhoW2V thật để đo latency
        freeze_syllable=arch["freeze_syllable"], freeze_contextual=arch["freeze_contextual"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, loader, vocab["tag"][scheme]


def _build_multihead(vocab, arch, checkpoint_path, data_dir, tokenizer, device, n_docs):
    from src.multihead_dataset import MultiHeadCollator, MultiHeadSpanDataset
    from src.multihead_model import BiLSTMMultiHeadCRFTagger

    test_ds = MultiHeadSpanDataset(data_dir / "test.json")
    if n_docs is not None:
        test_ds.docs = test_ds.docs[:n_docs]
    collate = MultiHeadCollator(vocab, tokenizer=tokenizer, use_contextual=arch["use_contextual"])
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate)

    model = BiLSTMMultiHeadCRFTagger(
        syllable_vocab_size=len(vocab["syllable"]), char_vocab_size=len(vocab["char"]),
        num_tags_aspect=len(vocab["tag"]["aspect"]), num_tags_polarity=len(vocab["tag"]["polarity"]),
        use_char=arch["use_char"], use_contextual=arch["use_contextual"],
        contextual_model_name=arch["contextual_model_name"], contextual_projected_dim=arch["contextual_projected_dim"],
        lstm_hidden=arch["lstm_hidden"], dropout=arch["dropout"],
        pretrained_syllable_matrix=None,
        freeze_syllable=arch["freeze_syllable"], freeze_contextual=arch["freeze_contextual"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--multihead-checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/hyperparams.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("UIT-ViSD4SA/iob"))
    parser.add_argument("--n-docs", type=int, default=None, help="Giới hạn số câu test dùng để đo (mặc định: toàn bộ tập test)")
    parser.add_argument("--no-contextual", action="store_true",
                         help="Tắt XLM-R -- PHẢI khớp đúng cấu hình lúc train 2 checkpoint (chỉ dùng khi checkpoint cũng được train với --no-contextual)")
    parser.add_argument("--lstm-hidden", type=int, default=None,
                         help="Override lstm_hidden -- PHẢI khớp đúng cấu hình lúc train 2 checkpoint")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.baseline_checkpoint is None and args.multihead_checkpoint is None:
        parser.error("Cần ít nhất 1 trong --baseline-checkpoint/--multihead-checkpoint")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    arch = dict(cfg["architecture"])
    if args.no_contextual:
        arch["use_contextual"] = False
    if args.lstm_hidden is not None:
        arch["lstm_hidden"] = args.lstm_hidden
    lat_cfg = cfg["inference_latency"]
    assert lat_cfg["batch_size"] == 1, "Script này CHỈ đo ở batch_size=1 theo đúng config -- xem docstring."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.data_dir / "vocab.json")
    tokenizer = AutoTokenizer.from_pretrained(arch["contextual_model_name"]) if arch["use_contextual"] else None

    results = {}

    if args.baseline_checkpoint is not None:
        model, loader, tag_vocab = _build_baseline(vocab, arch, args.baseline_checkpoint, args.data_dir, tokenizer, device, args.n_docs)
        results["baseline"] = measure_latency_baseline(
            model, loader, tag_vocab, device, n_warmup_batches=lat_cfg["n_warmup_batches"], use_amp=lat_cfg["use_amp"],
        )
        print(f"Baseline: {results['baseline']}")

    if args.multihead_checkpoint is not None:
        model, loader = _build_multihead(vocab, arch, args.multihead_checkpoint, args.data_dir, tokenizer, device, args.n_docs)
        results["multihead"] = measure_latency_multihead(
            model, loader, vocab["tag"]["aspect"], vocab["tag"]["polarity"], device,
            n_warmup_batches=lat_cfg["n_warmup_batches"], use_amp=lat_cfg["use_amp"],
        )
        print(f"Multihead: {results['multihead']}")

    if "baseline" in results and "multihead" in results:
        print()
        print_latency_comparison(results["baseline"], results["multihead"])

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
