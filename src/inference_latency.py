"""
Đo chi phí suy luận (latency) để trả lời trực tiếp bằng SỐ ĐO, không suy luận
lý thuyết: cấu hình M0 (`src/multihead_model.py::BiLSTMMultiHeadCRFTagger`,
2 CRF độc lập 21+7 nhãn) có không gian nhãn NHỎ HƠN baseline (`src/
bilstm_crf.py::BiLSTMCRFTagger`, 1 CRF 61 nhãn) ở bước Viterbi decode, NHƯNG
cần thêm bước hợp nhất (`merge_aspect_polarity_spans`, `src/
multihead_training.py`) mà baseline hoàn toàn không có -- 2 hiệu ứng kéo
ngược chiều nhau, không có cách nào biết trước độ trễ TỔNG tăng hay giảm mà
không đo thực tế.

Tách latency mỗi câu thành các giai đoạn RIÊNG BIỆT (đo bằng `time.
perf_counter()`, đồng bộ CUDA -- `torch.cuda.synchronize()` -- ngay trước/
sau mỗi giai đoạn để không đo nhầm thời gian kernel launch bất đồng bộ):

  - `model_forward_ms`: embedding fusion + BiLSTM (encoder, GIỐNG HỆT nhau
    giữa 2 cấu hình) + Viterbi decode CRF -- ĐÚNG giai đoạn "không gian
    nhãn nhỏ hơn" có thể giúp nhanh hơn (Viterbi mỗi bước thời gian O(T x
    L^2), L = số nhãn -- baseline L=61 trong 1 lần decode; M0 decode L=21
    rồi L=7 RIÊNG BIỆT, không phải L=28 gộp, nên rẻ hơn baseline theo lý
    thuyết -- cần xác nhận bằng số đo, không giả định).
  - `spans_decode_ms`: chuyển chuỗi tag id thành span (dict) -- baseline
    làm ĐÚNG 1 lần (`decode_doc_spans`, gọi `bio_to_spans` bên trong); M0
    làm việc NÀY 2 LẦN (1 lần/head) -- tự nó đã là 1 khoản M0 phải trả
    thêm, TRƯỚC CẢ bước hợp nhất, hay bị bỏ sót khi chỉ nhìn `model_
    forward_ms`.
  - `merge_ms`: CHỈ có ở M0 -- `merge_aspect_polarity_spans`, bước hoàn
    toàn không tồn tại ở baseline (đúng phần "bước gộp thêm chi phí" cần
    đo). Luôn 0.0 ở baseline (giữ field để 2 kết quả cùng shape, so sánh
    trực tiếp được bằng `print_latency_comparison`).
  - `total_ms`/`docs_per_sec`: tổng cả 3 giai đoạn trên -- con số QUYẾT
    ĐỊNH cuối cùng "độ trễ tăng hay giảm", không phải suy luận riêng từ
    `model_forward_ms`.

`n_warmup_batches` (mặc định 2) loại các batch ĐẦU khỏi số đo -- lần forward
đầu tiên qua XLM-R tốn thêm thời gian biên dịch kernel CUDA/cuDNN autotune,
không đại diện cho latency ổn định của các câu sau.
"""
from __future__ import annotations

import time

import torch

from src.evaluation import decode_doc_spans
from src.span_detection import bio_to_spans


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _tokens_with_offsets(tokens: list[str], token_offsets: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    return [(tok, start, end) for tok, (start, end) in zip(tokens, token_offsets)]


def measure_latency_baseline(
    model, loader, tag_vocab, device: torch.device,
    n_warmup_batches: int = 2, n_timed_batches: int | None = None, use_amp: bool = True,
) -> dict:
    """Đo latency baseline (`BiLSTMCRFTagger`, 1 CRF 61 nhãn -- scheme
    `aspect_polarity`). `tag_vocab` = `vocab["tag"][scheme]` (khớp đúng
    cách `src/training.py::predict_dataset` decode span). `merge_ms` LUÔN
    0.0 -- không có bước hợp nhất nào ở baseline."""
    from src.training import move_batch_to_device

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    batches = list(loader)
    if n_timed_batches is not None:
        batches = batches[: n_warmup_batches + n_timed_batches]

    n_docs = 0
    forward_time = spans_time = 0.0

    for bi, batch in enumerate(batches):
        batch_dev = move_batch_to_device(batch, device)
        timed = bi >= n_warmup_batches

        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            tag_paths = model.predict(batch_dev)
        _sync(device)
        t1 = time.perf_counter()

        for i in range(len(tag_paths)):
            n = batch["mask"][i].sum().item()
            toks = _tokens_with_offsets(batch["tokens"][i], batch["token_offsets"][i])
            decode_doc_spans(toks, tag_paths[i], tag_vocab)
        _sync(device)
        t2 = time.perf_counter()

        if timed:
            forward_time += t1 - t0
            spans_time += t2 - t1
            n_docs += len(batch["doc_ids"])

    total_time = forward_time + spans_time
    return {
        "config": "baseline (1 CRF, 61 nhãn, không hợp nhất)",
        "n_docs_timed": n_docs,
        "model_forward_ms_per_doc": forward_time / n_docs * 1000 if n_docs else 0.0,
        "spans_decode_ms_per_doc": spans_time / n_docs * 1000 if n_docs else 0.0,
        "merge_ms_per_doc": 0.0,
        "total_ms_per_doc": total_time / n_docs * 1000 if n_docs else 0.0,
        "docs_per_sec": n_docs / total_time if total_time else 0.0,
    }


def measure_latency_multihead(
    model, loader, aspect_tag_vocab, polarity_tag_vocab, device: torch.device,
    n_warmup_batches: int = 2, n_timed_batches: int | None = None, use_amp: bool = True,
) -> dict:
    """Đo latency M0 (`BiLSTMMultiHeadCRFTagger`, 2 CRF độc lập 21+7 nhãn) --
    tách riêng `spans_decode_ms` (2x `bio_to_spans`, TƯƠNG ĐƯƠNG bước
    baseline cũng phải làm, chỉ nhân đôi) và `merge_ms` (`merge_aspect_
    polarity_spans`, bước CHỈ M0 mới cần) để biết rõ từng phần cộng thêm
    bao nhiêu, không gộp chung thành 1 con số "hậu xử lý" mập mờ."""
    from src.multihead_training import merge_aspect_polarity_spans, move_batch_to_device

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    batches = list(loader)
    if n_timed_batches is not None:
        batches = batches[: n_warmup_batches + n_timed_batches]

    n_docs = 0
    forward_time = spans_time = merge_time = 0.0

    for bi, batch in enumerate(batches):
        batch_dev = move_batch_to_device(batch, device)
        timed = bi >= n_warmup_batches

        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            aspect_paths, polarity_paths = model.predict(batch_dev)
        _sync(device)
        t1 = time.perf_counter()

        per_doc_spans = []
        for i in range(len(aspect_paths)):
            toks = _tokens_with_offsets(batch["tokens"][i], batch["token_offsets"][i])
            aspect_tags = [aspect_tag_vocab.decode(t) for t in aspect_paths[i]]
            polarity_tags = [polarity_tag_vocab.decode(t) for t in polarity_paths[i]]
            aspect_spans = bio_to_spans(toks, aspect_tags)
            polarity_spans = bio_to_spans(toks, polarity_tags)
            per_doc_spans.append((aspect_spans, polarity_spans))
        _sync(device)
        t2 = time.perf_counter()

        for aspect_spans, polarity_spans in per_doc_spans:
            merge_aspect_polarity_spans(aspect_spans, polarity_spans)
        _sync(device)
        t3 = time.perf_counter()

        if timed:
            forward_time += t1 - t0
            spans_time += t2 - t1
            merge_time += t3 - t2
            n_docs += len(batch["doc_ids"])

    total_time = forward_time + spans_time + merge_time
    return {
        "config": "M0 multi-head (2 CRF độc lập, 21+7 nhãn, CÓ hợp nhất)",
        "n_docs_timed": n_docs,
        "model_forward_ms_per_doc": forward_time / n_docs * 1000 if n_docs else 0.0,
        "spans_decode_ms_per_doc": spans_time / n_docs * 1000 if n_docs else 0.0,
        "merge_ms_per_doc": merge_time / n_docs * 1000 if n_docs else 0.0,
        "total_ms_per_doc": total_time / n_docs * 1000 if n_docs else 0.0,
        "docs_per_sec": n_docs / total_time if total_time else 0.0,
    }


def print_latency_comparison(baseline_result: dict, multihead_result: dict, log_fn=print) -> None:
    """In bảng so sánh từng giai đoạn + kết luận RÀNH MẠCH: độ trễ TỔNG mỗi
    câu của M0 so với baseline tăng hay giảm bao nhiêu %, không dừng lại ở
    việc so sánh riêng `model_forward_ms` (dễ gây hiểu lầm rằng "nhãn nhỏ
    hơn = luôn nhanh hơn", bỏ qua đúng chi phí bước hợp nhất mà câu hỏi gốc
    yêu cầu phải tính vào)."""
    b, m = baseline_result, multihead_result
    log_fn(f"{'Giai đoạn':<32}{'Baseline (ms/câu)':>20}{'M0 (ms/câu)':>16}{'Chênh lệch':>16}")
    for key, label in [
        ("model_forward_ms_per_doc", "Model forward (encoder+CRF decode)"),
        ("spans_decode_ms_per_doc", "Giải mã tag -> span"),
        ("merge_ms_per_doc", "Hợp nhất 2 CRF (CHỈ M0 có)"),
        ("total_ms_per_doc", "TỔNG"),
    ]:
        bv, mv = b[key], m[key]
        diff = mv - bv
        log_fn(f"{label:<32}{bv:>20.3f}{mv:>16.3f}{diff:>+15.3f}ms")

    pct_change = (m["total_ms_per_doc"] - b["total_ms_per_doc"]) / b["total_ms_per_doc"] * 100 if b["total_ms_per_doc"] else 0.0
    log_fn(f"\nThông lượng: baseline={b['docs_per_sec']:.2f} câu/giây | M0={m['docs_per_sec']:.2f} câu/giây")
    if pct_change > 0:
        log_fn(f"\n=> M0 CHẬM HƠN baseline {pct_change:.1f}% về tổng độ trễ/câu -- bước hợp nhất "
               f"({m['merge_ms_per_doc']:.3f}ms/câu) ăn hết (và vượt) phần tiết kiệm được ở bước "
               f"decode nhờ không gian nhãn nhỏ hơn ({b['model_forward_ms_per_doc']:.3f} -> "
               f"{m['model_forward_ms_per_doc']:.3f}ms/câu).")
    elif pct_change < 0:
        log_fn(f"\n=> M0 NHANH HƠN baseline {abs(pct_change):.1f}% về tổng độ trễ/câu -- phần tiết "
               f"kiệm được ở bước decode (không gian nhãn nhỏ hơn) VẪN LỚN HƠN chi phí bước hợp "
               f"nhất cộng thêm ({m['merge_ms_per_doc']:.3f}ms/câu).")
    else:
        log_fn("\n=> M0 và baseline có tổng độ trễ/câu XẤP XỈ BẰNG NHAU.")
