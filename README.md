# CRF Multi-Head — Span Detection for Vietnamese ABSA (UIT-ViSD4SA)

Code for **"Decoupled CRF Architecture with a Shared Encoder for Span Detection in Vietnamese
Aspect-Based Sentiment Analysis"**: a shared encoder (syllable + character CharLSTM + XLM-R-large)
feeding two independent CRF heads (aspect + polarity) for span detection in Vietnamese aspect-based
sentiment analysis (ABSA), on the UIT-ViSD4SA dataset — extending Nguyen et al. (2021), PACLIC 35,
*"Span Detection for Aspect-Based Sentiment Analysis in Vietnamese"*.

## Architecture

```
Input sentence (syllable + char + XLM-R)
        │
   Embedding fusion (syllable PhoW2V 100d + char-BiLSTM 100d + XLM-R-large projected to 100d)
        │
   BiLSTM (400 dims/direction)
        │
        ▼
  ┌─────────────┐         ┌──────────────────┐
  │ CRF aspect   │         │ CRF polarity     │
  │ (21 tags)    │         │ (7 tags)         │
  └──────┬───────┘         └────────┬─────────┘
         └───────────┬──────────────┘
                      ▼
        merge_aspect_polarity_spans
```

## Repo structure

```
config/hyperparams.yaml   Hyperparameters (matches tab:hyperparams-shared)
scripts/
  prepare_data.py         Converts raw UIT-ViSD4SA -> IOB + vocab
  train.py                CLI to train the proposed model
  evaluate.py              Re-evaluates a checkpoint (Exact Match F1 + merge stats)
  measure_latency.py       Measures inference latency at batch_size=1
  smoke_test_multihead_model.py   Quick test (CPU, no GPU/real data needed)
src/                       Model + training + data + evaluation
UIT-ViSD4SA/iob/vocab.json  Pre-built vocab (contains no raw text)
```

## Installation

```bash
git clone <this repo's URL>
cd multihead-crf-uit-visd4sa
pip install -r requirements.txt
```

## Data

**UIT-ViSD4SA**: 35,396 manually-labeled spans over 11,122 Vietnamese phone-review comments, 10
aspects × 3 polarities. Source: https://github.com/kimkim00/UIT-ViSD4SA

This repo does not commit the converted data (`UIT-ViSD4SA/iob/{train,dev,test}.json`) — third-party
data, citation required. Only `vocab.json` is committed.

```bash
git clone https://github.com/kimkim00/UIT-ViSD4SA.git UIT-ViSD4SA-raw
python scripts/prepare_data.py --raw-dir UIT-ViSD4SA-raw/data --out-dir UIT-ViSD4SA/iob
```

Also needs **PhoW2V** (pretrained syllable embedding, ~458MB, https://github.com/datquocnguyen/PhoW2V)
extracted into a directory, pointed to via `--phow2v-dir` when running `scripts/train.py`.

**Citation required** if using the UIT-ViSD4SA data:
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

## Training / evaluation / latency

```bash
# Train (5 fixed seeds, hyperparameters from config/hyperparams.yaml)
python scripts/train.py --seeds 42 123 777 2024 2025

# Re-evaluate a checkpoint (Exact Match F1 + default-assignment/orphan rate)
python scripts/evaluate.py --checkpoint checkpoint_multihead_seed42.pt

# Measure inference latency at batch_size=1
python scripts/measure_latency.py --checkpoint checkpoint_multihead_seed42.pt
```

`train.py` prints macro-F1/micro-F1 on the test set directly, to compare against the paper's reported
**45.63% ± 0.74 macro-F1, 59.84% ± 0.77 micro-F1** (requires running all 5 seeds on GPU to confirm —
see `--help` on each script for `--seeds`/`--epochs`/`--no-contextual`).

## Testing

```bash
python scripts/smoke_test_multihead_model.py
```
No GPU/network required — checks `merge_aspect_polarity_spans`, `BiLSTMMultiHeadCRFTagger`'s
forward/backward pass, and (if data is available) one training + inference round on real data.

## Notes on reproduction

The model always runs the full `epochs` from the config (no early stopping). Data converted via
`scripts/prepare_data.py` may differ slightly (by one document) from the original data used to
produce the reported results (missing a manual offset-fix step from the original project, not
included in this repo).

## License & Citation

Source code: [LICENSE](LICENSE) (MIT). Citing this repo: [CITATION.cff](CITATION.cff). The
UIT-ViSD4SA data and PhoW2V belong to their original authors — see the Data section above.
