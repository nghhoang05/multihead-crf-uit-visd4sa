"""
BiLSTM-CRF with embedding fusion for UIT-ViSD4SA span detection, reproducing
the architecture of Nguyen et al. (2021, PACLIC 35) "Span Detection for
Aspect-Based Sentiment Analysis in Vietnamese":

    [syllable embedding (100d) | char-BiLSTM embedding (100d) | XLM-R embedding, projected to 100d]
        --> BiLSTM(hidden=400/direction) --> Linear --> CRF

(all three embedding sources are fine-tuned by default -- see below. All
three are 100-dimensional, matching Section 5.1's "embedding dimension of
100" for syllable/character/contextual alike -- XLM-R's native hidden_size,
768 for base / 1024 for large, is reduced to 100 via a Linear projection in
`ContextualEncoder`, see its docstring.)

Design notes / deviations from the paper (agreed with the user beforehand,
see the conversation log and notebook 08's intro):
  - Syllable embedding is initialized from PhoW2V's pretrained syllable-level
    Word2Vec vectors (see `src/pretrained_syllable_embedding.py`) and then
    FINE-TUNED together with the rest of the model (freeze_syllable=False by
    default) -- the paper's own source (baomoi.zip, Nguyen et al. 2017) is a
    dead link (verified), PhoW2V is the closest available substitute (same
    author lineage, larger 20GB corpus, syllable-level, dim=100 matches the
    paper's Section 5.1). This also matches that paper's own procedure for
    initializing syllable embeddings: "We initialize word and syllable
    embeddings with 100-dimensional pretrained embeddings, then learn them
    together with other model parameters during training" (Nguyen et al.,
    2017, Section 3.4) -- i.e. fine-tuning, not freezing, is the more
    faithful choice here. An earlier run froze this embedding instead (kept
    available via freeze_syllable=True) and scored *below* the pre-PhoW2V
    random-init baseline (35.12% vs 40.55% F1-macro, aspect_polarity) --
    switched to fine-tuning after that result. Character embeddings are
    still trained from scratch (no pretrained source exists or is expected
    for char-level).
  - XLM-R is FINE-TUNED by default (freeze_contextual=False), not used as a
    frozen feature extractor -- this was a deliberately deferred decision
    (the paper doesn't state whether XLM-R was fine-tuned) revisited once
    the syllable-embedding fix alone wasn't tested in isolation long enough
    to be conclusive, and because a frozen multi-hundred-million-parameter
    encoder contributing only generic representations was suspected to cap
    performance the same way frozen PhoW2V did. Fine-tuning XLM-R needs a
    MUCH smaller learning rate than the rest of the model or it destroys
    the pretrained weights in a few steps -- `src/training.py::train_model`
    uses a separate low-LR parameter group for exactly this reason. The
    frozen behavior is still available via freeze_contextual=True for
    comparison/ablation.
  - CRF is a from-scratch, dependency-light implementation (no torchcrf/
    flair) since those failed to install cleanly in this environment.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PAD, UNK = "<PAD>", "<UNK>"


# ---------------------------------------------------------------------------
# Character-level BiLSTM encoder: char ids (per syllable token) -> 1 vector
# ---------------------------------------------------------------------------
class CharEncoder(nn.Module):
    def __init__(self, char_vocab_size: int, char_emb_dim: int = 50, out_dim: int = 100, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_idx)
        assert out_dim % 2 == 0, "out_dim must be even (bidirectional concat)"
        self.lstm = nn.LSTM(char_emb_dim, out_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, char_ids: torch.Tensor, char_lengths: torch.Tensor) -> torch.Tensor:
        """
        char_ids: (batch*seq_len, max_char_len) int64 -- one row per syllable
                  token (batch and sequence dims flattened together).
        char_lengths: (batch*seq_len,) actual char count per token (>=1).
        Returns: (batch*seq_len, out_dim) -- one vector per token.
        """
        embedded = self.embedding(char_ids)
        lengths_clamped = char_lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths_clamped, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_directions, batch*seq_len, out_dim//2) -> concat fwd/bwd
        return torch.cat([h_n[0], h_n[1]], dim=-1)


# ---------------------------------------------------------------------------
# XLM-R contextual encoder (fine-tuned by default), mean-pooled from
# subwords back to syllables
# ---------------------------------------------------------------------------
class ContextualEncoder(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base", freeze: bool = False, projected_dim: int | None = 100):
        """`freeze=True` reproduces the earlier frozen-feature-extractor
        ablation (kept for comparison); default is fine-tune, since freezing
        both pretrained embedding sources (syllable + XLM-R) empirically
        underperformed the original random-init-syllable baseline (see
        `EmbeddingFusion`'s docstring).

        `projected_dim` (default 100): Section 5.1 of the paper states "Our
        word embeddings have three parts: syllable (1), character (2), and
        contextual from XLM-R (3), with an embedding dimension of 100" --
        read literally, all three parts (including XLM-R) are 100-dim, not
        XLM-R's native hidden_size (768 for base, 1024 for large). An
        earlier version of this code skipped this and concatenated XLM-R's
        raw hidden_size unprojected -- fixed here with a `Linear(hidden_size,
        projected_dim)` applied after mean-pooling to syllable level (a
        Linear commutes with the alignment-matrix mean-pool, so applying it
        after pooling instead of before is mathematically equivalent and
        cheaper: one projection per syllable instead of per subword).
        Pass `projected_dim=None` to restore the old unprojected behavior."""
        super().__init__()
        from transformers import AutoModel

        self.xlmr = AutoModel.from_pretrained(model_name)
        hidden_size = self.xlmr.config.hidden_size
        self.projection = nn.Linear(hidden_size, projected_dim) if projected_dim is not None else None
        self.out_dim = projected_dim if projected_dim is not None else hidden_size
        self.freeze = freeze
        if freeze:
            for p in self.xlmr.parameters():
                p.requires_grad = False
            self.xlmr.eval()
        else:
            # Fine-tuning a full transformer is memory-hungry (backward needs
            # every layer's activations); gradient checkpointing recomputes
            # them during backward instead of storing them, trading ~20-30%
            # more compute time for a large activation-memory reduction --
            # necessary headroom on a 16GB T4 (see training.py's CUDA OOM
            # note). No effect when frozen (no backward pass through XLM-R
            # in that case, so nothing to checkpoint).
            self.xlmr.gradient_checkpointing_enable()

    def encode_subwords(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns last_hidden_state (batch, subword_len, hidden_size).
        Frozen -> eval() + no_grad (matches inference-only behavior exactly
        as before). Fine-tuning -> follows the module's current train()/
        eval() mode (so XLM-R's own dropout is active during training) and
        builds a normal gradient graph."""
        if self.freeze:
            self.xlmr.eval()
            with torch.no_grad():
                out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def forward(self, input_ids, attention_mask, alignment: torch.Tensor) -> torch.Tensor:
        """
        alignment: (batch, seq_len, subword_len) float 0/1 mean-pooling matrix
                   mapping each syllable token to the subwords it overlaps
                   (rows sum to 1, or all-zero for padding tokens).
        Returns: (batch, seq_len, out_dim) contextual embedding per syllable
                 (out_dim = projected_dim if set, else XLM-R's native hidden_size).
        """
        subword_vecs = self.encode_subwords(input_ids, attention_mask)  # (B, S, H)
        pooled = torch.bmm(alignment, subword_vecs)  # (B, L, S) x (B, S, H) -> (B, L, H)
        return self.projection(pooled) if self.projection is not None else pooled


# ---------------------------------------------------------------------------
# Embedding fusion: syllable + char + contextual, concatenated
# ---------------------------------------------------------------------------
class EmbeddingFusion(nn.Module):
    def __init__(
        self,
        syllable_vocab_size: int,
        char_vocab_size: int,
        use_char: bool = True,
        use_contextual: bool = True,
        syllable_emb_dim: int = 100,
        char_out_dim: int = 100,
        contextual_model_name: str = "xlm-roberta-base",
        contextual_projected_dim: int | None = 100,
        pad_idx: int = 0,
        pretrained_syllable_matrix: torch.Tensor | None = None,
        freeze_syllable: bool = False,
        freeze_contextual: bool = False,
    ):
        """
        `pretrained_syllable_matrix`, if given (shape (syllable_vocab_size,
        syllable_emb_dim), see `src/pretrained_syllable_embedding.py`),
        initializes the syllable embedding from PhoW2V instead of random
        weights. `freeze_syllable` (default False -- fine-tune, matching
        Nguyen et al. 2017's own procedure for syllable embeddings, and
        the empirical result that freezing it underperformed even the
        random-init baseline) sets `requires_grad=False` on it when True,
        for the rare case you want to reproduce that frozen ablation.
        `freeze_contextual` (default False -- fine-tune XLM-R too) is
        forwarded to `ContextualEncoder`; when fine-tuning, use a much
        smaller learning rate for these params than for the rest of the
        model (see `src/training.py::train_model`'s `xlmr_lr`), or the
        pretrained XLM-R weights will be destroyed within a few steps.
        `contextual_projected_dim` (default 100) is forwarded to
        `ContextualEncoder` -- see its docstring for why (Section 5.1 of
        the paper).
        """
        super().__init__()
        self.syllable_embedding = nn.Embedding(syllable_vocab_size, syllable_emb_dim, padding_idx=pad_idx)
        if pretrained_syllable_matrix is not None:
            assert pretrained_syllable_matrix.shape == (syllable_vocab_size, syllable_emb_dim), (
                f"pretrained_syllable_matrix shape {tuple(pretrained_syllable_matrix.shape)} != "
                f"expected ({syllable_vocab_size}, {syllable_emb_dim})"
            )
            self.syllable_embedding.weight.data.copy_(pretrained_syllable_matrix)
        if freeze_syllable:
            self.syllable_embedding.weight.requires_grad = False
        self.use_char = use_char
        self.use_contextual = use_contextual
        self.output_dim = syllable_emb_dim

        if use_char:
            self.char_encoder = CharEncoder(char_vocab_size, out_dim=char_out_dim, pad_idx=pad_idx)
            self.output_dim += char_out_dim
        if use_contextual:
            self.contextual_encoder = ContextualEncoder(
                contextual_model_name, freeze=freeze_contextual, projected_dim=contextual_projected_dim,
            )
            self.output_dim += self.contextual_encoder.out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        bsz, seq_len = batch["syllable_ids"].shape
        parts = [self.syllable_embedding(batch["syllable_ids"])]

        if self.use_char:
            char_vecs = self.char_encoder(batch["char_ids"].view(bsz * seq_len, -1), batch["char_lengths"].view(-1))
            parts.append(char_vecs.view(bsz, seq_len, -1))

        if self.use_contextual:
            ctx_vecs = self.contextual_encoder(batch["subword_ids"], batch["subword_mask"], batch["alignment"])
            parts.append(ctx_vecs)

        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Linear-chain CRF (from scratch): forward algorithm (train) + Viterbi (decode)
# ---------------------------------------------------------------------------
class CRF(nn.Module):
    def __init__(self, num_tags: int, pad_idx: int | None = 0):
        """
        `pad_idx`, if given, is structurally forbidden as an emitted tag: a
        large constant penalty is added to that tag's emission score at
        every position (a non-trainable buffer, not a Parameter, so it can't
        drift back during training). Without this, an under-trained model
        can legitimately "predict" PAD as if it were a real tag, since
        nothing otherwise excludes it from the argmax/logsumexp.
        """
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.01)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)

        penalty = torch.zeros(num_tags)
        if pad_idx is not None:
            penalty[pad_idx] = -10000.0
        self.register_buffer("pad_penalty", penalty)

    def _mask_pad(self, emissions: torch.Tensor) -> torch.Tensor:
        return emissions + self.pad_penalty

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """emissions: (B, L, T), mask: (B, L) bool. Returns log partition (B,)."""
        emissions = self._mask_pad(emissions)
        bsz, seq_len, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]  # (B, T)
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (B, T_prev, 1)
            broadcast_emit = emissions[:, t].unsqueeze(1)  # (B, 1, T_cur)
            next_score = broadcast_score + self.transitions + broadcast_emit  # (B, T_prev, T_cur)
            next_score = torch.logsumexp(next_score, dim=1)  # (B, T_cur)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = self._mask_pad(emissions)
        bsz, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, seq_len):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + (trans + emit) * mask[:, t]
        last_tag_idx = mask.sum(1).long() - 1
        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_z = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return (log_z - gold_score).mean()

    def _forward_vars(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Same recursion as `_forward_alg`, but returns the score vector AT
        EVERY position instead of only the final log-partition -- i.e.
        alpha_t(k) = log sum over all tag prefixes of length t+1 ending in
        tag k at position t, of exp(start_transitions[y0] + emissions_0(y0)
        + ... + transitions[y_{t-1},k] + emissions_t(k)). Used by
        `marginal_probabilities` (added for `src/bamibert_cascade_model.py`'s
        soft aspect->polarity conditioning -- see that module's docstring).
        Returns (B, L, T)."""
        emissions = self._mask_pad(emissions)
        bsz, seq_len, _ = emissions.shape
        alphas = []
        score = self.start_transitions + emissions[:, 0]
        alphas.append(score)
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, t].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            alphas.append(score)
        return torch.stack(alphas, dim=1)

    def _backward_vars(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """beta_t(k) = log sum over all tag suffixes from position t+1 to
        each example's own real end, of exp(transitions[k,y_{t+1}] +
        emissions_{t+1}(y_{t+1}) + ... + end_transitions[y_last]) --
        deliberately EXCLUDES emissions_t(k) itself (that's alpha_t's job;
        see `marginal_probabilities`, which needs each emission counted
        exactly once: alpha_t(k) + beta_t(k) = total score of every path
        through tag k at position t).

        Right-padded, variable-length batches make this trickier than
        `_forward_vars`: beta's base case (`end_transitions`) belongs at
        EACH EXAMPLE'S OWN last real position, not at a fixed physical
        index. Handled with an explicit per-position injection, verified in
        `scripts/smoke_test_bamibert_cascade.py` against brute-force
        enumeration and against the identity logsumexp_k(alpha_t(k) +
        beta_t(k)) == log_Z for every real t (both must hold if this is
        correct -- the latter holds for ANY valid t simultaneously, a much
        stronger check than "the numbers happen to look reasonable")."""
        emissions = self._mask_pad(emissions)
        bsz, seq_len, num_tags = emissions.shape
        end_transitions_expanded = self.end_transitions.unsqueeze(0).expand(bsz, -1)

        betas = [None] * seq_len
        betas[seq_len - 1] = end_transitions_expanded.clone()  # correct base case for examples whose real length == seq_len; garbage-but-harmless placeholder for shorter ones (overwritten below via is_last_here)
        for t in range(seq_len - 2, -1, -1):
            next_score = self.transitions.unsqueeze(0) + emissions[:, t + 1].unsqueeze(1) + betas[t + 1].unsqueeze(1)
            computed = torch.logsumexp(next_score, dim=2)  # (B, T) indexed by the tag AT position t
            is_last_here = mask[:, t] & ~mask[:, t + 1]  # position t is THIS example's own last real position
            is_padding_here = ~mask[:, t]  # position t is past this example's real content entirely (unused, just carried forward)
            betas[t] = torch.where(
                is_last_here.unsqueeze(1), end_transitions_expanded,
                torch.where(is_padding_here.unsqueeze(1), betas[t + 1], computed),
            )
        return torch.stack(betas, dim=1)

    def marginal_probabilities(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """P(tag=k at position t | whole sequence), for every real position,
        via forward-backward: exp(alpha_t(k) + beta_t(k) - log_Z). Returns
        (B, L, T) -- values at padding positions are NOT meaningful
        probabilities (not renormalized, no guarantee of summing to 1) and
        must be masked out by the caller if used downstream; this
        deliberately does NOT apply a safety `softmax` here, so that a
        masking/off-by-one bug at REAL positions would surface as "doesn't
        sum to 1" in tests instead of being silently papered over."""
        log_z = self._forward_alg(emissions, mask)  # (B,), reuses the exact same method training already trusts
        alphas = self._forward_vars(emissions, mask)
        betas = self._backward_vars(emissions, mask)
        return torch.exp(alphas + betas - log_z.view(-1, 1, 1))

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding. Returns a list (len=batch) of best tag-id sequences
        (each already truncated to that sequence's real, unpadded length)."""
        emissions = self._mask_pad(emissions)
        bsz, seq_len, num_tags = emissions.shape
        history = []
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            next_score = broadcast_score + self.transitions
            best_score, best_idx = next_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_idx)
        score = score + self.end_transitions

        seq_lens = mask.sum(1).long()
        best_paths = []
        for b in range(bsz):
            length = seq_lens[b].item()
            best_last = score[b].argmax().item()
            path = [best_last]
            for t in range(length - 2, -1, -1):
                best_last = history[t][b, best_last].item()
                path.append(best_last)
            path.reverse()
            best_paths.append(path)
        return best_paths


# ---------------------------------------------------------------------------
# Full tagger: EmbeddingFusion -> BiLSTM -> Linear -> CRF
# ---------------------------------------------------------------------------
class BiLSTMCRFTagger(nn.Module):
    def __init__(
        self,
        syllable_vocab_size: int,
        char_vocab_size: int,
        num_tags: int,
        use_char: bool = True,
        use_contextual: bool = True,
        contextual_model_name: str = "xlm-roberta-base",
        contextual_projected_dim: int | None = 100,
        lstm_hidden: int = 400,
        dropout: float = 0.33,
        pad_idx: int = 0,
        tag_pad_idx: int = 0,
        pretrained_syllable_matrix: torch.Tensor | None = None,
        freeze_syllable: bool = False,
        freeze_contextual: bool = False,
        crf_cls: type = None,
        fusion_cls: type = None,
    ):
        """`pad_idx` is the PAD index in the syllable/char vocabs (embedding
        padding_idx); `tag_pad_idx` is the PAD index in the TAG vocab (a
        separate id space) -- both are 0 by this project's `Vocab` convention
        (see `src/span_dataset.py`), but kept as distinct parameters since
        they need not coincide in general. `pretrained_syllable_matrix` /
        `freeze_syllable` / `freeze_contextual` / `contextual_projected_dim`
        are forwarded to `EmbeddingFusion` -- see its docstring.

        `crf_cls` (default `None` -> the plain `CRF` above): lets a subclass
        swap in a CRF variant with the SAME forward-algorithm/Viterbi-decode
        behavior but a different loss reduction -- e.g. `src/focal_crf.py::
        FocalCRF`, which adds a focal-reweighted NLL method without changing
        `_forward_alg`/`_score_sentence`/`decode` at all. Existing callers
        that don't pass this keep the exact same plain-`CRF` behavior as
        before this parameter was added.

        `fusion_cls` (default `None` -> the plain `EmbeddingFusion` above):
        same idea, for the embedding fusion -- lets a subclass swap in a
        fusion variant that accepts this SAME call signature but changes
        what happens INSIDE one branch, e.g. `src/char_cnn_fusion.py::
        CharCNNEmbeddingFusion` (CharCNN instead of CharLSTM for the
        character branch, same early-fusion concatenation strategy).
        Existing callers that don't pass this keep the exact same plain-
        `EmbeddingFusion` behavior as before this parameter was added."""
        super().__init__()
        self.embedding_fusion = (fusion_cls or EmbeddingFusion)(
            syllable_vocab_size, char_vocab_size,
            use_char=use_char, use_contextual=use_contextual,
            contextual_model_name=contextual_model_name,
            contextual_projected_dim=contextual_projected_dim, pad_idx=pad_idx,
            pretrained_syllable_matrix=pretrained_syllable_matrix,
            freeze_syllable=freeze_syllable,
            freeze_contextual=freeze_contextual,
        )
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            self.embedding_fusion.output_dim, lstm_hidden,
            batch_first=True, bidirectional=True,
        )
        self.hidden2tag = nn.Linear(lstm_hidden * 2, num_tags)
        self.crf = (crf_cls or CRF)(num_tags, pad_idx=tag_pad_idx)

    def _emissions(self, batch: dict) -> torch.Tensor:
        emb = self.embedding_fusion(batch)
        emb = self.dropout(emb)
        lstm_out, _ = self.bilstm(emb)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out)

    def loss(self, batch: dict) -> torch.Tensor:
        emissions = self._emissions(batch)
        return self.crf.neg_log_likelihood(emissions, batch["tag_ids"], batch["mask"])

    def predict(self, batch: dict) -> list[list[int]]:
        self.eval()
        with torch.no_grad():
            emissions = self._emissions(batch)
            return self.crf.decode(emissions, batch["mask"])
