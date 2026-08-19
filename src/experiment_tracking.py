"""
Giai đoạn 0.2 (lộ trình cải tiến pipeline) — bảng theo dõi ablation chuẩn, dùng
xuyên suốt mọi giai đoạn sau (1-5): mỗi lần chạy 1 cấu hình mới, nạp file kết
quả qua đây để so sánh nhất quán, thay vì đọc JSON tay mỗi lần.

Theo đúng nguyên tắc rủi ro #3/#4/#5 của lộ trình:
  - Luôn hiện cả 3 mức nhãn nếu có nhiều file (aspect/polarity/aspect_polarity).
  - Luôn tách riêng F1 các lớp cực hiếm (DESIGN/PRICE/STORAGE/NEUTRAL) --
    KHÔNG chỉ nhìn macro-F1 tổng, vì macro-F1 có thể cải thiện chỉ nhờ các lớp
    vốn đã dễ (BATTERY/CAMERA#POSITIVE...) trong khi lớp hiếm vẫn F1=0%.
  - Luôn hiện chi phí tính toán (thời gian train, số tham số) cạnh F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Bảng 3 bài báo gốc (syllable+char+XLM-R-Large, cấu hình tốt nhất) -- mốc so sánh cố định.
PAPER_F1_MACRO = {"aspect": 0.6276, "polarity": 0.4977, "aspect_polarity": 0.4570}

# Các lớp cực hiếm cần theo dõi riêng xuyên suốt mọi giai đoạn (rủi ro #4 của lộ trình).
RARE_CLASS_SUBSTRINGS = ("NEUTRAL", "PRICE", "STORAGE", "DESIGN")


def load_experiment(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _total_train_time_min(result: dict) -> float:
    return sum(h.get("train_time_sec", 0.0) for h in result.get("history", [])) / 60


def _total_oom_skipped(result: dict) -> int:
    return sum(h.get("n_oom_skipped", 0) for h in result.get("history", []))


def is_rare_class(label: str) -> bool:
    return any(sub in label for sub in RARE_CLASS_SUBSTRINGS)


def summarize_experiments(experiments: dict[str, str | Path]) -> pd.DataFrame:
    """
    `experiments`: {label: path_to_results_json}, e.g.
        {"baseline (v8)": "results_aspect_polarity_final_pipeline.json",
         "+ CharCNN": "results_aspect_polarity_charcnn.json"}

    Returns one row per experiment: scheme, F1 micro/macro (dev best + test),
    gap so với bài báo, thời gian train, số tham số, số batch bị bỏ qua vì OOM,
    và F1-macro trung bình riêng trên các lớp hiếm (không tính vào macro-F1
    tổng để không bị các lớp dễ che lấp).
    """
    rows = []
    for label, path in experiments.items():
        result = load_experiment(path)
        cfg = result.get("config", {})
        scheme = cfg.get("scheme") or _infer_scheme_from_per_class(result.get("test_per_class", {}))
        per_class = result.get("test_per_class", {})
        rare = {k: v for k, v in per_class.items() if is_rare_class(k)}
        rare_f1s = [v["f1"] for v in rare.values()]
        n_zero_f1_rare = sum(1 for f1 in rare_f1s if f1 == 0.0)

        paper_f1 = PAPER_F1_MACRO.get(scheme)
        test_macro_f1 = result.get("test_macro", {}).get("f1")
        rows.append({
            "label": label,
            "scheme": scheme,
            "epochs_ran": cfg.get("epochs_ran"),
            "best_dev_macro_f1": result.get("best_dev_macro_f1"),
            "test_micro_f1": result.get("test_micro", {}).get("f1"),
            "test_macro_f1": test_macro_f1,
            "paper_macro_f1": paper_f1,
            "gap_vs_paper": (test_macro_f1 - paper_f1) if (test_macro_f1 is not None and paper_f1) else None,
            "rare_class_macro_f1": sum(rare_f1s) / len(rare_f1s) if rare_f1s else None,
            "n_rare_classes_f1_zero": n_zero_f1_rare,
            "n_rare_classes_total": len(rare_f1s),
            "train_time_min": round(_total_train_time_min(result), 1),
            "n_oom_skipped_total": _total_oom_skipped(result),
            "n_total_params": cfg.get("n_total_params"),
            "n_trainable_params": cfg.get("n_trainable_params"),
        })
    return pd.DataFrame(rows)


def _infer_scheme_from_per_class(per_class: dict) -> str:
    if not per_class:
        return "unknown"
    sample = next(iter(per_class))
    return "aspect_polarity" if "#" in sample else "unknown"


def rare_class_report(result_or_path: dict | str | Path) -> pd.DataFrame:
    """Per-class breakdown restricted to the rare classes tracked by risk #4
    of the roadmap (DESIGN/PRICE/STORAGE/*NEUTRAL). Use this instead of eyeballing
    the full 30-row per_class table every time."""
    result = result_or_path if isinstance(result_or_path, dict) else load_experiment(result_or_path)
    per_class = result.get("test_per_class", {})
    rows = [{"label": k, **v} for k, v in per_class.items() if is_rare_class(k)]
    df = pd.DataFrame(rows)
    return df.sort_values("f1") if not df.empty else df
