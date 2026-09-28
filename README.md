# CRF Multi-Head — Span Detection cho ABSA tiếng Việt (UIT-ViSD4SA)

Code kèm bài báo **D044**: mô hình đề xuất dùng shared encoder (syllable + character CharLSTM +
XLM-R-large) và 2 nhánh CRF độc lập (aspect + polarity) cho bài toán span detection trong phân tích
cảm xúc theo khía cạnh (ABSA) tiếng Việt, trên bộ dữ liệu UIT-ViSD4SA — mở rộng từ Nguyen et al.
(2021), PACLIC 35, *"Span Detection for Aspect-Based Sentiment Analysis in Vietnamese"*.

## Kiến trúc

```
Input câu (syllable + char + XLM-R)
        │
   Embedding fusion (syllable PhoW2V 100d + char-BiLSTM 100d + XLM-R-large chiếu 100d)
        │
   BiLSTM (400 chiều/hướng)
        │
        ▼
  ┌─────────────┐         ┌──────────────────┐
  │ CRF aspect   │         │ CRF polarity     │
  │ (21 nhãn)    │         │ (7 nhãn)         │
  └──────┬───────┘         └────────┬─────────┘
         └───────────┬──────────────┘
                      ▼
        merge_aspect_polarity_spans
```

## Cấu trúc repo

```
config/hyperparams.yaml   Siêu tham số (khớp tab:hyperparams-shared)
scripts/
  prepare_data.py         Chuyển UIT-ViSD4SA gốc -> IOB + vocab
  train.py                CLI huấn luyện mô hình đề xuất
  evaluate.py              Đánh giá lại 1 checkpoint (Exact Match F1 + merge-stats)
  measure_latency.py       Đo độ trễ suy luận batch_size=1
  smoke_test_multihead_model.py   Kiểm thử nhanh (CPU, không cần GPU/data thật)
src/                       Model + training + data + evaluation
UIT-ViSD4SA/iob/vocab.json  Vocab đã dựng sẵn (KHÔNG chứa văn bản gốc)
```

## Cài đặt

```bash
git clone <URL repo này>
cd multihead-crf-uit-visd4sa
pip install -r requirements.txt
```

## Dữ liệu

**UIT-ViSD4SA**: 35.396 span đã gán nhãn thủ công trên 11.122 bình luận điện thoại tiếng Việt, 10
khía cạnh × 3 cực tính. Nguồn: https://github.com/kimkim00/UIT-ViSD4SA

Repo này không commit dữ liệu đã chuyển đổi (`UIT-ViSD4SA/iob/{train,dev,test}.json`) — dữ liệu của
bên thứ ba, chỉ yêu cầu trích dẫn khi dùng. Chỉ `vocab.json` được commit sẵn.

```bash
git clone https://github.com/kimkim00/UIT-ViSD4SA.git UIT-ViSD4SA-raw
python scripts/prepare_data.py --raw-dir UIT-ViSD4SA-raw/data --out-dir UIT-ViSD4SA/iob
```

Cần thêm **PhoW2V** (syllable embedding pretrained, ~458MB, https://github.com/datquocnguyen/PhoW2V)
giải nén vào 1 thư mục, trỏ `--phow2v-dir` khi chạy `scripts/train.py`.

**Trích dẫn bắt buộc** nếu dùng dữ liệu UIT-ViSD4SA:
```bibtex
@inproceedings{thanh-etal-2021-span,
    title = "Span Detection for Aspect-Based Sentiment Analysis in Vietnamese",
    author = "Thanh, Kim Nguyen Thi and Khai, Sieu Huynh and Huynh, Phuc Pham and
              Luc, Luong Phan and Nguyen, Duc-Vu and Van, Kiet Nguyen",
    booktitle = "Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation",
    year = "2021", address = "Shanghai, China", publisher = "Association for Computational Lingustics",
    url = "https://aclanthology.org/2021.paclic-1.34", pages = "318--328",
}
```

## Chạy huấn luyện / đánh giá / đo latency

```bash
# Huấn luyện (5 seed cố định, đọc siêu tham số từ config/hyperparams.yaml)
python scripts/train.py --seeds 42 123 777 2024 2025

# Đánh giá lại 1 checkpoint (Exact Match F1 + default-assignment/orphan rate)
python scripts/evaluate.py --checkpoint checkpoint_multihead_seed42.pt

# Đo độ trễ suy luận, batch_size=1
python scripts/measure_latency.py --checkpoint checkpoint_multihead_seed42.pt
```

`train.py` in ra trực tiếp macro-F1/micro-F1 trên tập test, đối chiếu với **45.63% ± 0.74 macro-F1,
59.84% ± 0.77 micro-F1** đã báo cáo trong bài (cần chạy đủ 5 seed trên GPU để xác nhận lại số liệu
này — `--seeds`/`--epochs`/`--no-contextual` xem `--help` của từng script).

## Kiểm thử

```bash
python scripts/smoke_test_multihead_model.py
```
Không cần GPU/mạng — kiểm tra `merge_aspect_polarity_spans`, forward/backward của
`BiLSTMMultiHeadCRFTagger`, và (nếu đã có dữ liệu) 1 vòng huấn luyện + suy luận trên dữ liệu thật.

## Lưu ý khi tái lập

Mô hình LUÔN chạy đủ `epochs` trong config (mặc định 30), không early stopping. Dữ liệu chuyển đổi
bằng `scripts/prepare_data.py` có thể lệch nhẹ 1 tài liệu so với bản gốc dùng để báo cáo kết quả
(thiếu 1 bước sửa lỗi offset thủ công của dự án gốc, không nằm trong repo này).

## Giấy phép & Trích dẫn

Mã nguồn: [LICENSE](LICENSE) (MIT). Trích dẫn repo này: [CITATION.cff](CITATION.cff). Dữ liệu
UIT-ViSD4SA và PhoW2V thuộc bản quyền tác giả gốc — xem mục Dữ liệu ở trên.
