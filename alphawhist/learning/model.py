"""Small distributional action-value MLP for imperfect-information Whist.

For each of 52 physical cards, predict a categorical distribution over
0..7 ADDITIONAL tricks won by the acting player, conditioned on playing
that card now and a chosen future-play policy. Illegal cards receive no
probability and -inf expected value.

The model deliberately does not access Game, sampled hands, or hidden
opponent cards. It only consumes EncodedDecision / EncodedBatch arrays.

PyTorch is an OPTIONAL dependency: existing game/bot code does not import
this file. Install with `pip install -r requirements-ml.txt`.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from alphawhist.learning.encoding import (
    EncodedBatch, EncodedDecision, FEATURE_DIM, MAX_CARDS,
    MAX_ADDITIONAL_TRICKS,
)

NUM_TRICK_OUTCOMES = MAX_ADDITIONAL_TRICKS + 1


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class ActionPredictions:
    """Batch-first tensors: probabilities [B,52,8], values [B,52]."""
    probabilities: torch.Tensor
    expected_scores: torch.Tensor

    def best_card_ids(self) -> torch.Tensor:
        return self.expected_scores.argmax(dim=-1)


class DistributionalWhistMLP(nn.Module):
    """Fast shared-state MLP with one 8-bin categorical head per card."""

    def __init__(self, hidden_size: int = 128, residual_blocks: int = 2):
        super().__init__()
        if hidden_size < 8 or residual_blocks < 0:
            raise ValueError("Invalid MLP dimensions")
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            *(ResidualBlock(hidden_size) for _ in range(residual_blocks)),
        )
        self.head = nn.Linear(hidden_size, MAX_CARDS * NUM_TRICK_OUTCOMES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Raw logits [batch,52,8]; no legal or trick-count masking yet."""
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"Expected features [batch,{FEATURE_DIM}]")
        latent = self.encoder(features)
        return self.head(latent).reshape(-1, MAX_CARDS, NUM_TRICK_OUTCOMES)

    @staticmethod
    def _masked_logits(logits: torch.Tensor,
                       remaining_tricks: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[1:] != (MAX_CARDS, NUM_TRICK_OUTCOMES):
            raise ValueError("Expected logits [batch,52,8]")
        if remaining_tricks.shape != (logits.shape[0],):
            raise ValueError("Remaining trick count must be [batch]")
        if (remaining_tricks < 1).any() or (
            remaining_tricks > MAX_ADDITIONAL_TRICKS
        ).any():
            raise ValueError("Remaining tricks outside 1..7")
        k = torch.arange(NUM_TRICK_OUTCOMES, device=logits.device)
        return logits.masked_fill(
            k.view(1, 1, -1) > remaining_tricks[:, None, None],
            torch.finfo(logits.dtype).min,
        )

    def predict(self, features: torch.Tensor, legal_mask: torch.Tensor,
                remaining_tricks: torch.Tensor, tricks_won: torch.Tensor,
                bids: torch.Tensor) -> ActionPredictions:
        """Convert raw logits to legal action values using the exact score rule.

        Q(a) = tricks_already_won + E[additional_tricks | a]
               + 10 * P(additional_tricks == bid - tricks_already_won | a).
        """
        logits = self(features)
        batch = features.shape[0]
        if legal_mask.shape != (batch, MAX_CARDS) or legal_mask.dtype != torch.bool:
            raise ValueError("legal_mask must be bool [batch,52]")
        if not bool(legal_mask.any(dim=1).all()):
            raise ValueError("Every state must have at least one legal action")
        if tricks_won.shape != (batch,) or bids.shape != (batch,):
            raise ValueError("Trick counts and bids must be [batch]")

        logits = self._masked_logits(logits, remaining_tricks)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities * legal_mask.unsqueeze(-1)
        k = torch.arange(NUM_TRICK_OUTCOMES, device=features.device)
        final_tricks = tricks_won[:, None] + k[None, :]
        scores = final_tricks + 10.0 * (final_tricks == bids[:, None])
        expected_scores = (probabilities * scores[:, None, :]).sum(dim=-1)
        expected_scores = expected_scores.masked_fill(~legal_mask, float("-inf"))
        return ActionPredictions(probabilities, expected_scores)

    @torch.no_grad()
    def predict_encoded(self, encoded: EncodedDecision | EncodedBatch,
                        device: str | torch.device | None = None) -> ActionPredictions:
        """Convenient inference adapter for the NumPy information-state encoder."""
        if isinstance(encoded, EncodedDecision):
            encoded = EncodedBatch(
                encoded.features[None], encoded.legal_mask[None],
                __import__("numpy").asarray([encoded.remaining_tricks]),
                __import__("numpy").asarray([encoded.tricks_won]),
                __import__("numpy").asarray([encoded.bid]),
            )
        if not isinstance(encoded, EncodedBatch):
            raise TypeError("Expected EncodedDecision or EncodedBatch")
        if device is None:
            device = next(self.parameters()).device
        to_tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)
        was_training = self.training
        try:
            self.eval()
            return self.predict(
                to_tensor(encoded.features, torch.float32),
                to_tensor(encoded.legal_mask, torch.bool),
                to_tensor(encoded.remaining_tricks, torch.long),
                to_tensor(encoded.tricks_won, torch.long),
                to_tensor(encoded.bids, torch.long),
            )
        finally:
            self.train(was_training)


def distributional_loss(logits: torch.Tensor, target_probabilities: torch.Tensor,
                        legal_mask: torch.Tensor,
                        remaining_tricks: torch.Tensor) -> torch.Tensor:
    """Supervised soft-label cross entropy averaged across legal actions.

    Targets should be frequency histograms from rollouts, with bins 0..7.
    Illegal action rows are ignored, as are impossible trick-count bins.
    """
    if target_probabilities.shape != logits.shape:
        raise ValueError("Target shape must match logits [batch,52,8]")
    if legal_mask.shape != logits.shape[:2] or legal_mask.dtype != torch.bool:
        raise ValueError("legal_mask must be bool [batch,52]")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("Each sample needs a legal card")
    with torch.no_grad():
        sums = target_probabilities.sum(dim=-1)
        if not torch.allclose(sums[legal_mask],
                              torch.ones_like(sums[legal_mask]), atol=1e-4):
            raise ValueError("Each legal card target must sum to one")
        if bool((target_probabilities < 0).any()):
            raise ValueError("Target probabilities must be nonnegative")
        k = torch.arange(NUM_TRICK_OUTCOMES, device=logits.device)
        invalid = k[None, None, :] > remaining_tricks[:, None, None]
        if bool((target_probabilities.masked_select(
                invalid.expand_as(target_probabilities)) > 0).any()):
            raise ValueError("Target assigns probability to impossible trick count")
    masked = DistributionalWhistMLP._masked_logits(logits, remaining_tricks)
    per_action = -(target_probabilities * F.log_softmax(masked, dim=-1)).sum(dim=-1)
    return per_action.masked_select(legal_mask).mean()
