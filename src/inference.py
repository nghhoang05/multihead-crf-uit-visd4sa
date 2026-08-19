"""
End-to-end inference wrapper: raw Vietnamese TEXT in, (aspect, polarity,
aspect#polarity) SPANS out. Every processing step the paper's model needs
(syllable tokenization, syllable/char id lookup, XLM-R subword alignment,
BiLSTM-CRF forward, Viterbi decode, IOB-to-span decoding) happens inside
`SpanDetectionPipeline.predict` -- the caller never touches tokenization,
vocab, or embeddings directly.

Design decision -- ONE model, not three (see notebook 15 for the full
rationale): the paper treats `aspect`, `polarity`, and `aspect#polarity` as
three independent experiments (three rows of Table 3) to measure how label
granularity affects difficulty -- it does not specify how (or whether) to
combine them into a single deployable predictor. Running three independently
-trained taggers at inference time would be 3x the compute AND could yield
three different span boundaries for "the same" opinion, with no principled
way to reconcile them into one span carrying all three label views.

Since `aspect#polarity` already encodes both other views (e.g.
"BATTERY#POSITIVE" implies aspect=BATTERY, polarity=POSITIVE), this pipeline
trains and serves ONLY the aspect#polarity tagger, and DERIVES aspect/polarity
per span by splitting its label string. This keeps span boundaries internally
consistent (one decision per span) and is the cheaper, more defensible choice
for an actual text-in/spans-out pipeline.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.bilstm_crf import BiLSTMCRFTagger
from src.span_dataset import Collator, load_vocab
from src.span_detection import bio_to_spans, syllable_tokenize

SCHEME = "aspect_polarity"


class SpanDetectionPipeline:
    """Loads a trained aspect#polarity BiLSTM-CRF model and exposes
    `.predict(text)` / `.predict_batch(texts)` returning decoded spans."""

    def __init__(self, model: BiLSTMCRFTagger, vocab: dict, tokenizer, device: torch.device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.tag_vocab = vocab["tag"][SCHEME]
        self.collator = Collator(vocab, scheme=SCHEME, tokenizer=tokenizer, use_contextual=True)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        contextual_model_name: str = "xlm-roberta-large",
        device: str | None = None,
    ) -> "SpanDetectionPipeline":
        """Build a ready-to-use pipeline from a saved `BiLSTMCRFTagger.state_dict()`
        (as produced by `src/training.py::train_model`) and the vocab.json from
        notebook 08. This is the only "cold start" step -- everything after is
        just `.predict(text)`."""
        from transformers import AutoTokenizer

        device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        vocab = load_vocab(vocab_path)
        model = BiLSTMCRFTagger(
            syllable_vocab_size=len(vocab["syllable"]),
            char_vocab_size=len(vocab["char"]),
            num_tags=len(vocab["tag"][SCHEME]),
            use_char=True,
            use_contextual=True,
            contextual_model_name=contextual_model_name,
        )
        state_dict = torch.load(checkpoint_path, map_location=device_t)
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained(contextual_model_name)
        return cls(model, vocab, tokenizer, device_t)

    @torch.no_grad()
    def predict(self, text: str) -> list[dict]:
        """Single-text convenience wrapper around `predict_batch`."""
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[list[dict]]:
        """
        Returns, for each input text, a list of span dicts:
            {"text": "...", "start": int, "end": int,
             "aspect": "BATTERY", "polarity": "POSITIVE", "aspect_polarity": "BATTERY#POSITIVE"}
        Texts with no extractable tokens (empty/whitespace-only) yield [].
        """
        self.model.eval()

        examples, keep_idx = [], []
        for i, text in enumerate(texts):
            tokens = syllable_tokenize(text)
            if not tokens:
                continue
            examples.append({
                "doc_id": i, "text": text,
                "tokens": [t[0] for t in tokens],
                "token_offsets": [(t[1], t[2]) for t in tokens],
                "tags": ["O"] * len(tokens),  # placeholder: gold tags aren't needed for prediction
            })
            keep_idx.append(i)

        results: list[list[dict]] = [[] for _ in texts]
        if not examples:
            return results

        batch = self.collator(examples)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred_paths = self.model.predict(batch)

        for row, i in enumerate(keep_idx):
            text = texts[i]
            tok_texts = batch["tokens"][row]
            tok_offsets = batch["token_offsets"][row]
            token_tuples = [(tok_texts[j], tok_offsets[j][0], tok_offsets[j][1]) for j in range(len(tok_texts))]
            tags = [self.tag_vocab.decode(t) for t in pred_paths[row]]
            spans = bio_to_spans(token_tuples, tags)

            doc_result = []
            for s in spans:
                aspect, _, polarity = s["label"].partition("#")
                doc_result.append({
                    "text": text[s["start"]:s["end"]],
                    "start": s["start"], "end": s["end"],
                    "aspect": aspect, "polarity": polarity,
                    "aspect_polarity": s["label"],
                })
            results[i] = doc_result

        return results
