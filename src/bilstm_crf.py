"""
Shared encoder (embedding fusion) and CRF building blocks used by
`src/multihead_model.py::BiLSTMMultiHeadCRFTagger`, reproducing the
architecture of Nguyen et al. (2021, PACLIC 35) "Span Detection for
Aspect-Based Sentiment Analysis in Vietnamese" up through the BiLSTM step:

    [syllable embedding (100d) | char-BiLSTM embedding (100d) | XLM-R embedding, projected to 100d]
        --> BiLSTM(hidden=400/direction) --> Linear --> CRF

All three embedding sources are fine-tuned by default and are each
100-dimensional (Section 5.1). XLM-R's native hidden_size (768/1024) is
reduced to 100 via a Linear projection in `ContextualEncoder`.

Deviations from the paper: syllable embedding is initialized from PhoW2V
(the paper's own `baomoi.zip` source is a dead link) and fine-tuned, not
frozen -- freezing it empirically underperformed random-init embeddings
(35.12% vs 40.55% F1-macro). XLM-R is fine-tuned by default too (the paper
doesn't state whether it was). CRF is a from-scratch implementation (no
torchcrf/flair dependency).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PAD, UNK = "<PAD>", "<UNK>"


class CharEncoder(nn.Module):
    """Char ids (per syllable token) -> 1 vector, via a small char-level BiLSTM."""

    def __init__(self, char_vocab_size: int, char_emb_dim: int = 50, out_dim: int = 100, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_idx)
        assert out_dim % 2 == 0, "out_dim must be even (bidirectional concat)"
        self.lstm = nn.LSTM(char_emb_dim, out_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, char_ids: torch.Tensor, char_lengths: torch.Tensor) -> torch.Tensor:
        """char_ids: (batch*seq_len, max_char_len). char_lengths: (batch*seq_len,).
        Returns (batch*seq_len, out_dim)."""
        embedded = self.embedding(char_ids)
        lengths_clamped = char_lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths_clamped, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return torch.cat([h_n[0], h_n[1]], dim=-1)


class ContextualEncoder(nn.Module):
    """XLM-R (fine-tuned by default), mean-pooled from subwords back to syllables."""

    def __init__(self, model_name: str = "xlm-roberta-base", freeze: bool = False, projected_dim: int | None = 100):
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
            # Gradient checkpointing: trades compute for the activation memory
            # fine-tuning a full transformer needs (fits a 16GB T4).
            self.xlmr.gradient_checkpointing_enable()

    def encode_subwords(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.xlmr.eval()
            with torch.no_grad():
                out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def forward(self, input_ids, attention_mask, alignment: torch.Tensor) -> torch.Tensor:
        """alignment: (batch, seq_len, subword_len) mean-pooling matrix mapping each
        syllable to the subwords it overlaps. Returns (batch, seq_len, out_dim)."""
        subword_vecs = self.encode_subwords(input_ids, attention_mask)  # (B, S, H)
        pooled = torch.bmm(alignment, subword_vecs)  # (B, L, S) x (B, S, H) -> (B, L, H)
        return self.projection(pooled) if self.projection is not None else pooled


class EmbeddingFusion(nn.Module):
    """Syllable + char + contextual embeddings, concatenated."""

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
        """`pretrained_syllable_matrix` (see `src/pretrained_syllable_embedding.py`)
        initializes the syllable embedding from PhoW2V instead of random weights."""
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


class CRF(nn.Module):
    """Linear-chain CRF from scratch: forward algorithm (train) + Viterbi (decode)."""

    def __init__(self, num_tags: int, pad_idx: int | None = 0):
        """`pad_idx`, if given, gets a large constant penalty added to its emission
        score at every position, so an under-trained model can't "predict" PAD."""
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

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding. Returns a list (len=batch) of best tag-id sequences,
        each truncated to that sequence's real, unpadded length."""
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
