# CRF Multi-Head — Span Detection cho ABSA tiếng Việt (UIT-ViSD4SA)

Code kèm bài báo **D044**: mô hình đề xuất dùng **shared encoder** (syllable + character CharLSTM +
XLM-R-large) và **hai nhánh CRF độc lập** (aspect + polarity) cho bài toán span detection trong phân
tích cảm xúc theo khía cạnh (ABSA) tiếng Việt, trên bộ dữ liệu UIT-ViSD4SA — mở rộng từ Nguyen et
al. (2021), PACLIC 35, *"Span Detection for Aspect-Based Sentiment Analysis in Vietnamese"*.

> Repo này chỉ chứa **mô hình đề xuất** (2 nhánh CRF độc lập: aspect + polarity).

## 1. Kiến trúc

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
        (tie-break: giữ span polarity gặp ĐẦU TIÊN
         trong vòng lặp, so sánh `>` nghiêm ngặt,
         overlap tính theo offset KÝ TỰ)
```

## 2. Cấu trúc repo

```
config/hyperparams.yaml            Siêu tham số (khớp tab:hyperparams-shared)
notebooks/19_multihead_crf_model.ipynb   Notebook Colab (tiện lợi, KHÔNG phải nguồn tái lập chính thức)
scripts/
  prepare_data.py                  Chuyển UIT-ViSD4SA gốc -> IOB + vocab
  train.py                         CLI huấn luyện mô hình đề xuất
  evaluate.py                      Đánh giá lại 1 checkpoint (Exact Match F1 + merge-stats)
  measure_latency.py               Đo độ trễ suy luận batch_size=1 (Section IV-B)
  results_to_csv.py                Gộp nhiều results_*.json (đa-seed) thành 1 CSV
  table_per_aspect_f1.py           Bảng F1 theo khía cạnh, mean +- std qua nhiều seed
  smoke_test_multihead_model.py    Kiểm thử nhanh (CPU, không cần GPU/data thật)
src/                               Model + training + data + evaluation
UIT-ViSD4SA/iob/vocab.json          Vocab đã dựng sẵn (KHÔNG chứa văn bản gốc)
```

## 3. Cài đặt

```bash
git clone <URL repo này>
cd multihead-crf-uit-visd4sa
pip install -r requirements.txt
```

## 4. Dữ liệu

**UIT-ViSD4SA**: 35.396 span đã gán nhãn thủ công trên 11.122 bình luận điện thoại tiếng Việt, 10
khía cạnh × 3 cực tính. Nguồn: https://github.com/kimkim00/UIT-ViSD4SA

Repo này **không commit dữ liệu đã chuyển đổi** (`UIT-ViSD4SA/iob/{train,dev,test}.json`) — dữ liệu
của bên thứ ba, không có giấy phép redistribute rõ ràng, chỉ yêu cầu trích dẫn. Chỉ `vocab.json`
(thống kê suy ra, không chứa văn bản gốc) được commit sẵn.

```bash
git clone https://github.com/kimkim00/UIT-ViSD4SA.git UIT-ViSD4SA-raw
python scripts/prepare_data.py --raw-dir UIT-ViSD4SA-raw/data --out-dir UIT-ViSD4SA/iob
```

**Lưu ý về độ chính xác của bước chuyển đổi**: script trên tái hiện quy trình gốc (whitespace
syllable-tokenize, gán IOB, dựng vocab) nhưng **không** áp dụng 1 bước sửa lỗi offset thủ công riêng
của dự án gốc (122 span, thực hiện trong 1 notebook không có trong repo này) — số liệu tái lập có
thể lệch nhẹ (đã đo: lệch đúng 1 tài liệu ở tập train/dev so với thống kê 7.784/1.113 trong README
gốc của UIT-ViSD4SA, tập test khớp chính xác 2.225). Nếu cần khớp tuyệt đối 100% với dữ liệu đã dùng
để có số liệu trong bài, cần dùng đúng file `train.json`/`dev.json`/`test.json` gốc của dự án chính
(không có trong repo standalone này).

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

Notebook có tải thêm **PhoW2V** (syllable embedding pretrained, ~458MB) -- chỉ dùng cho mục đích
nghiên cứu/giáo dục; khi báo cáo kết quả cần trích dẫn Nguyen, Dao, Nguyen (2020), *"A Pilot Study of
Text-to-SQL Semantic Parsing for Vietnamese"*, Findings of ACL: EMNLP 2020.

## 5. Tái lập từng bảng kết quả

### Bảng III/IV (Exact Match F1, macro/micro, 5 seed cố định)

```bash
# PhoW2V: tải + giải nén vào phow2v/extracted/ trước (xem notebooks/19 Mục 3, hoặc gdown thủ công)
python scripts/train.py --seeds 42 123 777 2024 2025

python scripts/results_to_csv.py --glob "results_multihead_seed*.json" --out table_multihead.csv
```

In ra trực tiếp mean ± std qua 5 seed (macro-F1, micro-F1) — đối chiếu với **45.63% ± 0.74 macro-F1,
59.84% ± 0.77 micro-F1** đã báo cáo trong bài (số liệu này CẦN chạy thật trên GPU để xác nhận lại —
xem mục 7 "Giới hạn" dưới đây).

Đánh giá lại 1 checkpoint đã có (không train lại), gồm cả default-assignment rate/orphan rate
(Section III-C):
```bash
python scripts/evaluate.py --checkpoint checkpoint_multihead_seed42.pt
```

### Section IV-B (độ trễ suy luận, batch_size=1)

```bash
python scripts/measure_latency.py --checkpoint checkpoint_multihead_seed42.pt
```

In riêng từng giai đoạn (model forward, giải mã span, hợp nhất 2 CRF) — cho biết bước hợp nhất
chiếm bao nhiêu % tổng độ trễ/câu.

### Bảng V (F1 theo khía cạnh, mean ± std qua 5 seed)

```bash
python scripts/table_per_aspect_f1.py --glob "results_multihead_seed*.json" --out table_v.csv
```

### (Tuỳ chọn) Notebook Colab

`notebooks/19_multihead_crf_model.ipynb` huấn luyện mô hình đề xuất trên Colab (GPU T4) — tiện cho
khám phá tương tác, nhưng **`scripts/train.py` là nguồn tái lập chính thức**. Sửa `GITHUB_REPO_URL`
ở Mục 2 của notebook thành URL repo thật sau khi push.

## 6. Kiểm thử

```bash
python scripts/smoke_test_multihead_model.py
```
Không cần GPU/mạng — kiểm tra `merge_aspect_polarity_spans`, forward/backward của
`BiLSTMMultiHeadCRFTagger`, và (nếu đã có dữ liệu) 1 vòng huấn luyện + suy luận trên dữ liệu thật.

## 7. Giới hạn / lưu ý khi tái lập (đọc trước khi báo cáo lại số liệu)

- **Không có early stopping**: mô hình LUÔN chạy đủ `epochs` trong config (mặc định 30), không dừng
  sớm — **cần bạn tự xác nhận lại với bản thảo bài báo** xem mục Huấn luyện có nói rõ có/không dùng
  early stopping hay không; nếu bài báo có nêu, cần khôi phục lại patience trong
  `src/multihead_training.py::train_model` cho khớp.
- **`requirements.txt` không được xác nhận bit-for-bit** đúng phiên bản đã dùng trên Google Colab
  lúc tạo ra số liệu báo cáo (Colab không lưu log version) — xem lưu ý ngay trong file đó.
- **`scripts/prepare_data.py`** tái tạo dữ liệu IOB từ file jsonl gốc nhưng thiếu 1 bước sửa lỗi
  offset thủ công của dự án chính (xem mục 4) — lệch 1 tài liệu ở train/dev.
- **Số liệu 45.63% ± 0.74 / 59.84% ± 0.77** (macro/micro-F1, cam kết với Reviewer 1) **chưa được
  xác nhận lại bằng 1 lần chạy GPU thật trong quá trình chuẩn bị repo này** — code đã qua kiểm thử
  đầy đủ (chạy đúng, không lỗi, trên CPU với cấu hình rút gọn), nhưng việc tái lập ĐÚNG con số cần
  chạy thật `scripts/train.py` với đầy đủ 5 seed trên GPU (nhiều giờ/seed với XLM-R-large).

## 8. Đối chiếu shared encoder với bài báo gốc (Nguyen et al. 2021)

*(Embedding fusion + BiLSTM, dùng chung bởi cả 2 nhánh CRF, kế thừa trực tiếp từ Nguyen et al. 2021
-- phần "2 CRF độc lập + merge" là đóng góp riêng của bài D044, không đối chiếu ở đây.)*

| Khoản mục | Khớp bài báo | Khác / diễn giải | Không có thông tin trong bài báo |
|---|---|---|---|
| Chiều embedding syllable/char (100d) | ✅ | | |
| Dropout 0.33 | ✅ | | |
| Tiêu chí đúng: exact-match (start+end+label) | ✅ | | |
| P/R/F1 micro + macro + per-class | ✅ | | |
| Tokenize theo âm tiết (syllable-level) | ✅ | | |
| weight_decay chỉ áp cho CRF (Eq. 11) | ✅ (phạm vi) | Giá trị `1e-2` là số AdamW tiêu chuẩn, không phải số bài báo | |
| Nguồn syllable embedding pretrained | | PhoW2V thay `baomoi.zip` gốc (đã chết link) | |
| Lớp chiếu XLM-R về 100 chiều | | Diễn giải nghĩa đen Section 5.1 | |
| "batch_size=5000" | | Diễn giải là ngân sách TOKEN, không phải số câu | |
| Fine-tune hay đóng băng syllable/XLM-R | | | ✅ (đã grep toàn văn "freeze/frozen/fine-tune" = 0 kết quả) |
| Optimizer, learning rate, learning rate riêng cho XLM-R | | | ✅ |
| Số epoch, tiêu chí dừng/chọn checkpoint | | | ✅ |
| Chạy đa-seed / báo cáo mean±std | | | ✅ |

## 9. Giấy phép

Mã nguồn: xem [LICENSE](LICENSE) — **cần điền tên chủ sở hữu bản quyền** (mặc định soạn sẵn MIT,
tham khảo quy định của trường/khoa trước khi công bố chính thức). Dữ liệu UIT-ViSD4SA và PhoW2V
thuộc bản quyền tác giả gốc tương ứng — xem mục 4 về yêu cầu trích dẫn.

## 10. Trích dẫn repo này

Xem [CITATION.cff](CITATION.cff) — **còn 1 số trường TODO cần điền** (tên bài báo chính xác, danh
sách tác giả, URL GitHub, DOI Zenodo) sau khi hoàn tất archive.
