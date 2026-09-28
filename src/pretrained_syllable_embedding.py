"""
Loads PhoW2V's pretrained syllable-level Word2Vec vectors
(word2vec_vi_syllables_100dims, see https://github.com/datquocnguyen/PhoW2V)
and builds an embedding matrix aligned with this project's syllable `Vocab`
(see `src/span_dataset.py`).

Usage:
    kv = load_word2vec_vectors(path_or_dir)
    matrix, stats = build_syllable_embedding_matrix(vocab["syllable"], kv)
    print(stats)  # coverage report
    # matrix is then handed to EmbeddingFusion(..., pretrained_syllable_matrix=matrix)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.vietnamese_tone_normalization import normalize_tone

PAD, UNK = "<PAD>", "<UNK>"
_VECTOR_FILE_PATTERNS = ("*.txt", "*.vec", "*.bin")


def _find_vector_file(path: str | Path) -> Path:
    """`path` may already be a vector file, or a directory containing one
    (e.g. the extracted PhoW2V zip). Picks the largest matching file if a
    directory is given, since the exact filename inside the zip isn't
    documented and may vary."""
    path = Path(path)
    if path.is_file():
        return path
    candidates = [f for pattern in _VECTOR_FILE_PATTERNS for f in path.rglob(pattern)]
    if not candidates:
        raise FileNotFoundError(
            f"No .txt/.vec/.bin vector file found under {path}. "
            "Pass the exact file path instead of a directory."
        )
    return max(candidates, key=lambda f: f.stat().st_size)


def load_word2vec_vectors(path: str | Path):
    """Returns a gensim KeyedVectors, trying text format first (PhoW2V ships
    as plain word2vec text format) then falling back to binary."""
    from gensim.models import KeyedVectors

    vec_path = _find_vector_file(path)
    try:
        return KeyedVectors.load_word2vec_format(str(vec_path), binary=False)
    except UnicodeDecodeError:
        return KeyedVectors.load_word2vec_format(str(vec_path), binary=True)


def _lookup(kv, token: str) -> np.ndarray | None:
    """Try, in order: exact token, lowercased, tone-normalized lowercased.
    PhoW2V's own corpus preprocessing lowercases and tone-normalizes, so the
    lowercased+tone-normalized form is the most likely match for anything
    not found verbatim."""
    for candidate in (token, token.lower(), normalize_tone(token.lower())):
        if candidate in kv:
            return kv[candidate]
    return None


def build_syllable_embedding_matrix(vocab, kv, freeze_note: bool = True) -> tuple[torch.Tensor, dict]:
    """vocab: this project's `Vocab` (src/span_dataset.py) built over syllable
    IOB training data. kv: gensim KeyedVectors from `load_word2vec_vectors`.

    Returns (embedding_matrix, stats). Row 0 (<PAD>) -> zeros; row 1 (<UNK>)
    -> mean of matched vectors; found rows -> pretrained vector; not-found
    rows -> left at random init."""
    dim = kv.vector_size
    n = len(vocab.itos)
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)

    found_vectors = []
    oov_examples = []
    for idx, token in enumerate(vocab.itos):
        if token in (PAD, UNK):
            continue
        vec = _lookup(kv, token)
        if vec is not None:
            matrix[idx] = vec
            found_vectors.append(vec)
        elif len(oov_examples) < 20:
            oov_examples.append(token)

    matrix[vocab.stoi[PAD]] = 0.0
    if found_vectors:
        matrix[vocab.stoi[UNK]] = np.mean(found_vectors, axis=0)

    n_real = n - 2  # exclude PAD/UNK from the coverage denominator
    n_found = len(found_vectors)
    stats = {
        "vocab_size": n,
        "vector_dim": dim,
        "n_real_tokens": n_real,
        "n_found": n_found,
        "n_oov": n_real - n_found,
        "coverage": n_found / n_real if n_real else 0.0,
        "oov_examples": oov_examples,
    }
    return torch.tensor(matrix, dtype=torch.float32), stats
