"""
BiLSTM + 2-independent-CRF-heads span detection model -- configuration #2 in
this project's 3-way span-detection ablation:
  #1 baseline:   `src/bilstm_crf.py::BiLSTMCRFTagger`      (1 CRF, 61-way combined `aspect#polarity` tag)
  #2 this file:  `BiLSTMMultiHeadCRFTagger`                (2 CRFs: aspect 21-way + polarity 7-way)
  #3 Span-ViSD:  `src/span_model.py::SpanEnumerationModel` (span-enumeration + pruning + 2 classification heads)

#2 exists to isolate the "multi-head" idea (asking 2 small, easy questions
instead of 1 big, hard one) as a SINGLE-VARIABLE ablation against the
baseline, instead of only being able to compare against #3 (Span-ViSD),
which changes BOTH the label space AND the whole span-finding mechanism
(enumeration + pruning instead of CRF decoding) at once -- a comparison
against #3 alone cannot tell you whether an observed F1 change is caused by
multi-head or by switching extraction paradigms.

Step 1 (`EmbeddingFusion`) and the extraction MECHANISM (CRF Viterbi
decoding over the shared BiLSTM hidden states) are IDENTICAL to the
baseline -- reused verbatim from `src/bilstm_crf.py`. The only difference is
splitting the 61-way combined `O,B-<aspect>#<polarity>,I-<aspect>#<polarity>`
tag space into two smaller, INDEPENDENTLY-decoded CRF heads (aspect:
`O,B-<aspect>,I-<aspect>` x10 = 21 tags; polarity:
`O,B-<polarity>,I-<polarity>` x3 = 7 tags) trained JOINTLY through the one
shared encoder. Joint training through a shared encoder is what lets the
polarity head share statistics across every aspect (e.g. NEUTRAL learned
from CAMERA#NEUTRAL, BATTERY#NEUTRAL, ... generalizes to DESIGN#NEUTRAL via
the shared polarity CRF's own weights) -- a single 61-way tag space cannot
do this, since B-DESIGN#NEUTRAL is its own independent, unshared parameter.

Because the two CRFs decode independently, they can (and will) disagree on
exactly where a span starts/ends. See `merge_aspect_polarity_spans` in
`src/multihead_training.py` for how the two decoded segmentations are
reconciled back into the single (start, end, "ASPECT#POLARITY") format that
`src/evaluation.py::evaluate` expects -- the SAME evaluation function used
to score the baseline and Span-ViSD, so all 3 configurations are directly
comparable.
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
        """Every parameter here has the exact same meaning/default as
        `src/bilstm_crf.py::BiLSTMCRFTagger` (see its docstring) -- the only
        additions are `num_tags_aspect`/`num_tags_polarity` replacing that
        model's single `num_tags`, since there are now two independent tag
        spaces instead of one combined one."""
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
        """Step 1, identical to the baseline: embedding fusion -> BiLSTM.
        Both CRF heads read from this SAME tensor -- this is the "shared
        encoder" that lets gradients from either loss term update
        representations the other head also benefits from."""
        emb = self.embedding_fusion(batch)
        emb = self.dropout(emb)
        lstm_out, _ = self.bilstm(emb)
        return self.dropout(lstm_out)

    def loss_components(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss_aspect, loss_polarity) UNSUMMED -- factored out of
        `loss()` so subclasses that reweight the two terms differently than
        a plain 1:1 sum (e.g. `src/multihead_uncertainty_model.py::
        BiLSTMMultiHeadUncertaintyTagger`, which learns a per-task weight
        instead of hardcoding equal weight) can reuse this exact forward
        pass instead of duplicating it. Purely additive: `loss()` below is
        unchanged in behavior/signature, still returns the same plain sum
        as before this method existed -- every existing caller (`src/
        multihead_training.py::train_model`, `scripts/smoke_test_multihead_
        model.py`) keeps working unmodified."""
        hidden = self._shared_hidden(batch)
        emissions_aspect = self.hidden2tag_aspect(hidden)
        emissions_polarity = self.hidden2tag_polarity(hidden)
        loss_aspect = self.crf_aspect.neg_log_likelihood(emissions_aspect, batch["tag_ids_aspect"], batch["mask"])
        loss_polarity = self.crf_polarity.neg_log_likelihood(emissions_polarity, batch["tag_ids_polarity"], batch["mask"])
        return loss_aspect, loss_polarity

    def loss(self, batch: dict) -> torch.Tensor:
        """Sum of both CRFs' negative log-likelihoods, unweighted (both are
        "equally important" sub-questions of the original combined tagging
        problem, unlike e.g. Span-ViSD's `mention_loss_weight`, which scales
        an auxiliary filtering objective against the main classification
        one -- here both heads ARE the main objective)."""
        loss_aspect, loss_polarity = self.loss_components(batch)
        return loss_aspect + loss_polarity

    def predict(self, batch: dict) -> tuple[list[list[int]], list[list[int]]]:
        """Returns (aspect_tag_paths, polarity_tag_paths) -- each head's own
        Viterbi decode, INDEPENDENTLY of the other (see this module's
        docstring for why they can disagree on boundaries, and
        `src/multihead_training.py::merge_aspect_polarity_spans` for how
        that disagreement is resolved into final spans)."""
        self.eval()
        with torch.no_grad():
            hidden = self._shared_hidden(batch)
            aspect_paths = self.crf_aspect.decode(self.hidden2tag_aspect(hidden), batch["mask"])
            polarity_paths = self.crf_polarity.decode(self.hidden2tag_polarity(hidden), batch["mask"])
        return aspect_paths, polarity_paths
