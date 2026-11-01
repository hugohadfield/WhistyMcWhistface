"""PyTorch CPU tests for the optional learned action-value model."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphawhist.core.InformationState import InformationState
from alphawhist.learning.encoding import (
    FEATURE_DIM, CARD_TO_ID, encode_decision, encode_batch,
)
from alphawhist.learning.model import DistributionalWhistMLP, distributional_loss


def state_with_bid(bid=1, cards=("14h", "2s")):
    state = InformationState.new_round(
        cards, 3, round_index=3, dealer=2, leader=0,
        cards_per_player=len(cards),
    )
    state = state.apply_public_event({
        "type": "trump", "player": 2, "suit": "h",
    })
    for player in range(3):
        state = state.apply_public_event({
            "type": "bid", "player": player, "bid": bid,
        })
    return state


def test_forward_batch_action_shape_and_gradients():
    model = DistributionalWhistMLP(hidden_size=32, residual_blocks=1)
    features = torch.randn(4, FEATURE_DIM, requires_grad=True)
    logits = model(features)
    assert logits.shape == (4, 52, 8)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert all(p.grad is not None for p in model.parameters())


def test_predict_respects_legal_and_remaining_trick_masks():
    torch.manual_seed(17)
    model = DistributionalWhistMLP(hidden_size=32, residual_blocks=1)
    states = (state_with_bid(), state_with_bid(cards=("14h", "9s", "2c")))
    batch = encode_batch(states)
    pred = model.predict_encoded(batch)
    assert pred.probabilities.shape == (2, 52, 8)
    assert pred.expected_scores.shape == (2, 52)
    expected = torch.as_tensor(batch.legal_mask)
    assert torch.all(pred.probabilities[~expected] == 0)
    assert torch.all(torch.isneginf(pred.expected_scores[~expected]))
    assert torch.all(torch.isfinite(pred.expected_scores[expected]))
    for i, limit in enumerate(batch.remaining_tricks):
        assert torch.all(pred.probabilities[i, :, limit + 1:] == 0)
        torch.testing.assert_close(
            pred.probabilities[i, expected[i], :].sum(dim=-1),
            torch.ones(int(expected[i].sum())),
        )
    best = pred.best_card_ids()
    for i in range(2):
        assert batch.legal_mask[i, best[i].item()]


def test_expected_score_applies_exact_bid_bonus():
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    state = encode_decision(state_with_bid(bid=1))
    # Override logits to make one card certain to win exactly one additional
    # trick, and the other certain to win zero.
    logits = torch.full((1, 52, 8), -50.0)
    a, b = CARD_TO_ID["14h"], CARD_TO_ID["2s"]
    logits[0, a, 1] = 50.0
    logits[0, b, 0] = 50.0
    model.forward = lambda features: logits
    result = model.predict_encoded(state)
    assert result.expected_scores[0, a].item() == pytest.approx(11.0)
    assert result.expected_scores[0, b].item() == pytest.approx(0.0)
    assert result.best_card_ids().tolist() == [a]


def test_distributional_training_loss_uses_only_legal_actions():
    torch.manual_seed(5)
    model = DistributionalWhistMLP(hidden_size=32, residual_blocks=1)
    encoded = encode_batch([state_with_bid(), state_with_bid(cards=("7c", "8h"))])
    features = torch.from_numpy(encoded.features)
    legal = torch.from_numpy(encoded.legal_mask)
    remaining = torch.from_numpy(encoded.remaining_tricks)
    target = torch.zeros((2, 52, 8))
    target[..., 0] = legal.to(torch.float32)
    logits = model(features)
    loss = distributional_loss(logits, target, legal, remaining)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0


def test_distributional_loss_rejects_impossible_or_bad_targets():
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    state = encode_decision(state_with_bid())
    feats = torch.as_tensor(state.features[None])
    legal = torch.as_tensor(state.legal_mask[None])
    remaining = torch.tensor([state.remaining_tricks])
    raw = model(feats)
    target = torch.zeros_like(raw)
    target[..., 0] = legal.float()
    with pytest.raises(ValueError, match="sum to one"):
        distributional_loss(raw, target * 0.5, legal, remaining)
    impossible = target.clone()
    impossible[..., 0] = 0
    impossible[..., 3] = legal.float()
    with pytest.raises(ValueError, match="impossible"):
        distributional_loss(raw, impossible, legal, remaining)


def test_model_state_dict_roundtrip(tmp_path):
    torch.manual_seed(333)
    model = DistributionalWhistMLP(hidden_size=32)
    batch = encode_decision(state_with_bid())
    expected = model.predict_encoded(batch).expected_scores
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = DistributionalWhistMLP(hidden_size=32)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    torch.testing.assert_close(restored.predict_encoded(batch).expected_scores, expected)


def test_invalid_dimensions_and_unusable_action_masks():
    with pytest.raises(ValueError):
        DistributionalWhistMLP(hidden_size=0)
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    with pytest.raises(ValueError):
        model(torch.zeros(1, FEATURE_DIM - 1))
    batch = encode_batch([state_with_bid()])
    with pytest.raises(ValueError, match="legal action"):
        model.predict(
            torch.from_numpy(batch.features),
            torch.zeros(1, 52, dtype=torch.bool),
            torch.from_numpy(batch.remaining_tricks),
            torch.from_numpy(batch.tricks_won),
            torch.from_numpy(batch.bids),
        )
