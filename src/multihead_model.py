"""
BiLSTM + 2-independent-CRF-heads span detection model. Instead of 1 big CRF
over a 61-way combined `aspect#polarity` tag space, ask 2 smaller questions
independently -- an aspect CRF (21 tags) and a polarity CRF (7 tags),
trained JOINTLY through one shared encoder (`src/bilstm_crf.py`).

Joint training through a shared encoder lets the polarity head share
statistics across every aspect (e.g. NEUTRAL learned from CAMERA#NEUTRAL
generalizes to DESIGN#NEUTRAL via the shared polarity CRF's weights) -- a
single 61-way tag space can't do this (B-DESIGN#NEUTRAL is its own
independent parameter).

Because the two CRFs decode independently, they can disagree on span
boundaries -- see `merge_aspect_polarity_spans` in
`src/multihead_training.py` for how that's reconciled.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.bilstm_crf import CRF, EmbeddingFusion


class BiLSTMMultiHeadCRFTagger(nn.Module):
    def __init__(
        self,
        syllable_vocab_size: int,
        char_vocab_size: int,
        num_tags_aspect: int,
        num_tags_polarity: int,
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
    ):
        super().__init__()
        self.embedding_fusion = EmbeddingFusion(
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
        self.hidden2tag_aspect = nn.Linear(lstm_hidden * 2, num_tags_aspect)
        self.hidden2tag_polarity = nn.Linear(lstm_hidden * 2, num_tags_polarity)
        self.crf_aspect = CRF(num_tags_aspect, pad_idx=tag_pad_idx)
        self.crf_polarity = CRF(num_tags_polarity, pad_idx=tag_pad_idx)

    def _shared_hidden(self, batch: dict) -> torch.Tensor:
        """Embedding fusion -> BiLSTM. Both CRF heads read from this same
        tensor -- the "shared encoder"."""
        emb = self.embedding_fusion(batch)
        emb = self.dropout(emb)
        lstm_out, _ = self.bilstm(emb)
        return self.dropout(lstm_out)

    def loss_components(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss_aspect, loss_polarity) unsummed."""
        hidden = self._shared_hidden(batch)
        emissions_aspect = self.hidden2tag_aspect(hidden)
        emissions_polarity = self.hidden2tag_polarity(hidden)
        loss_aspect = self.crf_aspect.neg_log_likelihood(emissions_aspect, batch["tag_ids_aspect"], batch["mask"])
        loss_polarity = self.crf_polarity.neg_log_likelihood(emissions_polarity, batch["tag_ids_polarity"], batch["mask"])
        return loss_aspect, loss_polarity

    def loss(self, batch: dict) -> torch.Tensor:
        """Sum of both CRFs' negative log-likelihoods, unweighted."""
        loss_aspect, loss_polarity = self.loss_components(batch)
        return loss_aspect + loss_polarity

    def predict(self, batch: dict) -> tuple[list[list[int]], list[list[int]]]:
        """Returns (aspect_tag_paths, polarity_tag_paths) -- each head's own
        Viterbi decode, independently of the other."""
        self.eval()
        with torch.no_grad():
            hidden = self._shared_hidden(batch)
            aspect_paths = self.crf_aspect.decode(self.hidden2tag_aspect(hidden), batch["mask"])
            polarity_paths = self.crf_polarity.decode(self.hidden2tag_polarity(hidden), batch["mask"])
        return aspect_paths, polarity_paths
