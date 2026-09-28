"""
Gộp nhiều file `results_<model>_seed<seed>.json` (từ `scripts/train.py`)
thành 1 file CSV tổng hợp -- KHÔNG cần checkpoint mô hình (nặng), chỉ cần
các file JSON kết quả nhẹ đã lưu, để người khác đối chiếu số liệu mà không
cần chạy lại từ đầu.

Cách dùng:
    python scripts/results_to_csv.py --glob "results_multihead_seed*.json" --out table_multihead.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glob", required=True, help='Mẫu glob cho các file results JSON, vd "results_multihead_seed*.json"')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"Không tìm thấy file nào khớp mẫu: {args.glob}")

    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        row = {
            "file": p,
            "model": r["config"]["model"],
            "seed": r["config"]["seed"],
            "epochs_ran": r["config"]["epochs_ran"],
            "best_dev_macro_f1": r["best_dev_macro_f1"],
            "test_micro_f1": r["test_micro"]["f1"],
            "test_micro_precision": r["test_micro"]["precision"],
            "test_micro_recall": r["test_micro"]["recall"],
            "test_macro_f1": r["test_macro"]["f1"],
            "test_macro_precision": r["test_macro"]["precision"],
            "test_macro_recall": r["test_macro"]["recall"],
            "train_time_min": r.get("train_time_min"),
        }
        if "merge_stats" in r:
            row["default_assigned_rate_pct"] = r["merge_stats"]["default_assigned_rate_pct"]
            row["orphan_polarity_discarded_rate_pct"] = r["merge_stats"]["orphan_polarity_discarded_rate_pct"]
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    micro_f1s = [r["test_micro_f1"] for r in rows]
    macro_f1s = [r["test_macro_f1"] for r in rows]
    n = len(rows)
    mean_micro = sum(micro_f1s) / n
    mean_macro = sum(macro_f1s) / n
    std_micro = (sum((x - mean_micro) ** 2 for x in micro_f1s) / (n - 1)) ** 0.5 if n > 1 else 0.0
    std_macro = (sum((x - mean_macro) ** 2 for x in macro_f1s) / (n - 1)) ** 0.5 if n > 1 else 0.0

    print(f"Đã gộp {n} file -> {args.out}")
    print(f"Micro-F1: {mean_micro * 100:.2f}% +- {std_micro * 100:.2f}pp (n={n} seed)")
    print(f"Macro-F1: {mean_macro * 100:.2f}% +- {std_macro * 100:.2f}pp (n={n} seed)")
    if n < 2:
        print("Lưu ý: cần ít nhất 2 seed để std có ý nghĩa.")


if __name__ == "__main__":
    main()
