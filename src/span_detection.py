"""
Syllable-level sequence-labeling utilities for reproducing the UIT-ViSD4SA
paper's span detection task (Nguyen et al., 2021, PACLIC 35):
"Span Detection for Aspect-Based Sentiment Analysis in Vietnamese".

The paper formalizes span detection as IOB sequence labeling at the
SYLLABLE level -- for written Vietnamese, whitespace already separates
syllables (a multi-syllable word is written as several space-separated
syllables), so syllable tokenization here is simple whitespace splitting.
This is deliberately NOT word-segmentation (e.g. underthesea.word_tokenize,
which merges syllables into compound words) -- that is a different token
granularity used elsewhere in this project's EDA notebooks, not what this
paper's model consumes.

Three label schemes are supported, matching the paper's three evaluated
label variants:
  - "aspect":          B-CAMERA, I-CAMERA, O, ...            (10 classes)
  - "polarity":        B-POSITIVE, I-POSITIVE, O, ...        (3 classes)
  - "aspect_polarity":  B-CAMERA#POSITIVE, I-CAMERA#POSITIVE, O, ...  (30 classes)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

Token = tuple[str, int, int]  # (text, char_start, char_end)

_TOKEN_RE = re.compile(r"\S+")


def syllable_tokenize(text: str) -> list[Token]:
    """Whitespace tokenization preserving each token's character offsets in `text`."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _span_label_for_scheme(aspect: str, polarity: str, scheme: str) -> str:
    if scheme == "aspect":
        return aspect
    if scheme == "polarity":
        return polarity
    if scheme == "aspect_polarity":
        return f"{aspect}#{polarity}"
    raise ValueError(f"Unknown scheme: {scheme}")


def spans_to_bio(tokens: list[Token], spans: list[dict], scheme: str = "aspect_polarity") -> list[str]:
    """
    Convert character-offset spans (each a dict with start/end/aspect/polarity)
    into a per-token IOB tag list aligned with `tokens`. A token is assigned to
    a span if it overlaps the span's [start, end) at all (standard char-span ->
    token-tag conversion rule); the first overlapping token gets a B- tag, the
    rest get I- tags.
    """
    tags = ["O"] * len(tokens)
    for span in spans:
        label = _span_label_for_scheme(span["aspect"], span["polarity"], scheme)
        first = True
        for i, (_, tok_start, tok_end) in enumerate(tokens):
            if tok_end > span["start"] and tok_start < span["end"]:
                tags[i] = f"{'B' if first else 'I'}-{label}"
                first = False
    return tags


def bio_to_spans(tokens: list[Token], tags: list[str]) -> list[dict]:
    """Decode a per-token IOB tag list back into character-offset spans
    (inverse of `spans_to_bio`) -- used both to validate round-trip
    conversion and to decode a trained model's predictions."""
    spans = []
    current = None
    for (_, tok_start, tok_end), tag in zip(tokens, tags):
        if tag == "O":
            if current:
                spans.append(current)
                current = None
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "B" or current is None or current["label"] != label:
            if current:
                spans.append(current)
            current = {"start": tok_start, "end": tok_end, "label": label}
        else:  # "I-" continuing the same label
            current["end"] = tok_end
    if current:
        spans.append(current)
    return spans


@dataclass
class TaggedDoc:
    doc_id: str
    text: str
    tokens: list[str]
    token_offsets: list[tuple[int, int]]
    tags: list[str]


def convert_document(doc: dict, scheme: str = "aspect_polarity") -> TaggedDoc:
    tokens = syllable_tokenize(doc["text"])
    spans = [
        {"start": s["start"], "end": s["end"], "aspect": s["aspect"], "polarity": s["polarity"]}
        for s in doc["spans"]
    ] if "spans" in doc else [
        {"start": s[0], "end": s[1], "aspect": s[2].split("#", 1)[0], "polarity": s[2].split("#", 1)[1]}
        for s in doc["labels"]
    ]
    tags = spans_to_bio(tokens, spans, scheme=scheme)
    return TaggedDoc(
        doc_id=doc["doc_id"], text=doc["text"],
        tokens=[t[0] for t in tokens], token_offsets=[(t[1], t[2]) for t in tokens],
        tags=tags,
    )


def convert_dataset(docs: list[dict], scheme: str = "aspect_polarity") -> list[TaggedDoc]:
    return [convert_document(d, scheme=scheme) for d in docs]
