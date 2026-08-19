# CRF Multi-Head — Span Detection cho ABSA tiếng Việt (UIT-ViSD4SA)

Tái hiện + mở rộng bài toán **span detection** cho phân tích cảm xúc theo khía cạnh (ABSA) tiếng
Việt, dựa trên *"Span Detection for Aspect-Based Sentiment Analysis in Vietnamese"* (Nguyen et al.,
PACLIC 35, 2021) và bộ dữ liệu **UIT-ViSD4SA**.

Đây là **cấu hình #2** trong 1 ablation 3 chiều rộng hơn (baseline 1-head CRF gộp / **multi-head 2
CRF độc lập (repo này)** / span-enumeration) — repo này tách riêng ĐÚNG 1 biến để cô lập hiệu ứng
"multi-head" (chia nhỏ 1 head 30-lớp `aspect#polarity` gộp thành 2 CRF nhỏ hơn, aspect + polarity),
không đổi cơ chế trích xuất span (vẫn CRF tuần tự + Viterbi decode).

## Kiến trúc

```
Input câu (syllable + char + XLM-R)
        │
   Embedding fusion (syllable PhoW2V + char-BiLSTM + XLM-R chiếu 100 chiều)
        │
   BiLSTM (400 chiều/hướng)
        │
   ┌────┴────┐
   ▼         ▼
CRF aspect  CRF polarity     <- 2 CRF ĐỘC LẬP, cùng đọc 1 encoder chung
(21 tag)    (7 tag)
   │         │
   └────┬────┘
        ▼
merge_aspect_polarity_spans   <- hợp nhất 2 chuỗi quyết định độc lập thành
                                   (start, end, "ASPECT#POLARITY")
```

Vì 2 CRF decode độc lập, chúng có thể cho biên span khác nhau — `src/multihead_training.py::
merge_aspect_polarity_spans` xử lý việc này (biên của CRF aspect làm chuẩn, chọn polarity theo
overlap lớn nhất, mặc định `POSITIVE` nếu không polarity span nào chồng lấn).

## Cấu trúc repo

```
notebooks/19_multihead_crf_model.ipynb   Notebook chính (Google Colab, GPU T4)
src/
  bilstm_crf.py                          Embedding fusion + CRF gốc (baseline, tái sử dụng)
  multihead_model.py                     BiLSTMMultiHeadCRFTagger (2 CRF độc lập)
  multihead_dataset.py                   Dataset/collate cho 2 CRF
  multihead_training.py                  Training loop + merge_aspect_polarity_spans
  span_dataset.py, span_detection.py     Tiện ích syllable-tokenize/IOB dùng chung
  pretrained_syllable_embedding.py       Nạp PhoW2V + xây ma trận embedding
  token_batch_sampler.py                 Dynamic token-budget batching
  evaluation.py                          Exact-span-match P/R/F1 (micro/macro/per-class)
  experiment_tracking.py                 So sánh nhiều lần chạy/cấu hình
  training.py                            Tiện ích chung (set_seed, param groups, XLM-R lr riêng)
  vietnamese_tone_normalization.py       Chuẩn hoá dấu tiếng Việt
  inference.py                           Suy luận trên câu tuỳ ý (sau khi có checkpoint)
scripts/
  prepare_data.py                        Chuyển UIT-ViSD4SA gốc -> IOB + vocab (Mục "Dữ liệu")
  smoke_test_multihead_model.py          Kiểm thử nhanh (CPU, không cần GPU/data thật)
UIT-ViSD4SA/iob/vocab.json                Vocab đã dựng sẵn (syllable/char/tag) -- KHÔNG chứa văn bản gốc
```

## Dữ liệu

**UIT-ViSD4SA**: 35.396 span gán nhãn thủ công trên 11.122 bình luận điện thoại tiếng Việt, theo 10
khía cạnh (BATTERY, CAMERA, DESIGN, FEATURES, GENERAL, PERFORMANCE, PRICE, SCREEN, SER&ACC, STORAGE)
× 3 cực tính (NEGATIVE, NEUTRAL, POSITIVE). Nguồn gốc: https://github.com/kimkim00/UIT-ViSD4SA

Repo này **không commit sẵn dữ liệu đã chuyển đổi** (`UIT-ViSD4SA/iob/{train,dev,test}.json`) — dữ
liệu là của bên thứ ba, không có giấy phép redistribute rõ ràng (chỉ yêu cầu trích dẫn). Chỉ
`vocab.json` (thống kê suy ra: danh sách âm tiết/ký tự/nhãn, không chứa văn bản bình luận gốc) được
commit sẵn.

**Notebook 19 tự động tải dữ liệu gốc + chuyển đổi khi chạy** (Mục 2) — không cần thao tác tay. Nếu
muốn tự chạy riêng:

```bash
git clone https://github.com/kimkim00/UIT-ViSD4SA.git
python scripts/prepare_data.py --raw-dir UIT-ViSD4SA/data --out-dir UIT-ViSD4SA/iob
```

Script tái hiện đúng quy trình chuyển đổi gốc (whitespace syllable-tokenize, gán IOB theo 3 biến
thể nhãn `aspect`/`polarity`/`aspect_polarity`, dựng vocab từ tập train) — xem docstring của
`scripts/prepare_data.py` để biết 1 khác biệt nhỏ đã biết so với dữ liệu dùng để báo cáo kết quả
gốc (không áp dụng 1 bước sửa lỗi offset thủ công riêng của dự án gốc).

**Trích dẫn bắt buộc** nếu dùng dữ liệu này:
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

Ngoài ra, notebook có tải **PhoW2V** (syllable embedding pretrained, ~458MB, mirror Google Drive
công khai của `datquocnguyen/PhoW2V`) — chỉ dùng cho mục đích nghiên cứu/giáo dục, không
redistribute file gốc; khi báo cáo kết quả cần trích dẫn Nguyen, Dao, Nguyen (2020), *"A Pilot
Study of Text-to-SQL Semantic Parsing for Vietnamese"*, Findings of ACL: EMNLP 2020.

## Chạy trên Google Colab (khuyến nghị)

1. Mở `notebooks/19_multihead_crf_model.ipynb` trên Colab (File → Open notebook → GitHub, dán URL
   repo này, hoặc mở trực tiếp từ trang GitHub bằng nút "Open in Colab" nếu bạn thêm badge).
2. **Runtime → Change runtime type → GPU (T4)**.
3. Sửa `GITHUB_REPO_URL` ở Mục 2 thành URL repo GitHub thật của bạn.
4. **Runtime → Run all** — Mục 2 tự `git clone` repo này + dữ liệu gốc, tự chuyển đổi dữ liệu nếu
   chưa có. Không cần upload file tay.

## Chạy cục bộ

```bash
git clone <URL repo này>
cd multihead-crf-uit-visd4sa
pip install -r requirements.txt
pip install transformers gensim gdown   # cần cho huấn luyện thật (không pin cứng phiên bản, xem requirements.txt)

git clone https://github.com/kimkim00/UIT-ViSD4SA.git UIT-ViSD4SA-raw
python scripts/prepare_data.py --raw-dir UIT-ViSD4SA-raw/data --out-dir UIT-ViSD4SA/iob

python scripts/smoke_test_multihead_model.py   # kiểm thử nhanh, CPU, ~vài giây
jupyter notebook notebooks/19_multihead_crf_model.ipynb   # huấn luyện thật cần GPU
```

## Kiểm thử

```bash
python scripts/smoke_test_multihead_model.py
```

Không cần GPU/mạng — kiểm tra `merge_aspect_polarity_spans`, forward/backward của
`BiLSTMMultiHeadCRFTagger`, và (nếu đã chạy `prepare_data.py`) 1 vòng huấn luyện + suy luận trên dữ
liệu thật.

## Kết quả tham chiếu

Bài báo gốc (syllable + char + XLM-R-Large, cấu hình tốt nhất): F1-macro (aspect_polarity) =
**45.70%**, aspect = 62.76%, polarity = 49.77% (Bảng 3). Notebook in bảng so sánh trực tiếp với các
con số này ở cell cuối.

## Giấy phép

Mã nguồn trong repo này được chia sẻ cho mục đích nghiên cứu/học thuật. Bộ dữ liệu UIT-ViSD4SA và
PhoW2V thuộc bản quyền của tác giả gốc tương ứng — xem mục "Dữ liệu" ở trên về yêu cầu trích dẫn.
