"""
Dựng lại bảng F1 theo từng khía cạnh (per-aspect), dạng mean +- std qua
nhiều seed, từ các file `results_<model>_seed<seed>.json` đã lưu -- không
cần chạy lại mạng nơ-ron. Hiện cả 2 mức chi tiết: theo từng lớp
`aspect#polarity` (30 lớp, đúng `test_per_class` trong file JSON) VÀ gộp lại
theo riêng khía cạnh (10 lớp, trung bình cộng F1 của 3 cực tính cùng khía
cạnh đó) -- dùng mức nào tuỳ theo đúng định dạng Bảng trong bài.

Cách dùng:
    python scripts/table_per_aspect_f1.py --glob "results_multihead_seed*.json" --out table_v.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"Không tìm thấy file nào khớp mẫu: {args.glob}")

    per_class_runs = defaultdict(list)   # "ASPECT#POLARITY" -> [f1_seed1, f1_seed2, ...]
    per_aspect_runs = defaultdict(list)  # "ASPECT" -> [f1_seed1_pol1, f1_seed1_pol2, ..., f1_seed2_pol1, ...]

    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        for label, m in r["test_per_class"].items():
            per_class_runs[label].append(m["f1"])
            aspect = label.split("#", 1)[0]
            per_aspect_runs[aspect].append(m["f1"])

    def mean_std(values):
        n = len(values)
        mean = sum(values) / n
        std = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5 if n > 1 else None
        return mean, std, n

    print(f"=== F1 theo aspect#polarity (30 lớp, {len(paths)} seed) ===")
    class_rows = []
    for label in sorted(per_class_runs):
        mean, std, n = mean_std(per_class_runs[label])
        class_rows.append({"label": label, "mean_f1_pct": mean * 100, "std_f1_pct": std * 100 if std is not None else "", "n_runs": n})
        std_str = f"{std * 100:.2f}" if std is not None else "n/a"
        print(f"  {label:<25} mean={mean * 100:6.2f}%  std={std_str}pp  (n={n})")

    print(f"\n=== F1 gộp theo khía cạnh (10 lớp, trung bình cộng 3 cực tính) ===")
    aspect_rows = []
    for aspect in sorted(per_aspect_runs):
        mean, std, n = mean_std(per_aspect_runs[aspect])
        aspect_rows.append({"label": aspect, "mean_f1_pct": mean * 100, "std_f1_pct": std * 100 if std is not None else "", "n_runs": n})
        std_str = f"{std * 100:.2f}" if std is not None else "n/a"
        print(f"  {aspect:<14} mean={mean * 100:6.2f}%  std={std_str}pp  (n={n})")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "mean_f1_pct", "std_f1_pct", "n_runs"])
        writer.writeheader()
        writer.writerow({"label": "--- per aspect#polarity (30 lớp) ---", "mean_f1_pct": "", "std_f1_pct": "", "n_runs": ""})
        writer.writerows(class_rows)
        writer.writerow({"label": "--- per aspect, gộp cực tính (10 lớp) ---", "mean_f1_pct": "", "std_f1_pct": "", "n_runs": ""})
        writer.writerows(aspect_rows)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
