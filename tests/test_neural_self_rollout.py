"""Regression tests for information-safe neural continuation of OUR turns."""
import contextlib
import io
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphawhist.core.FastRollout import CARD_ID, SimState
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.checkpoint import save_checkpoint
from alphawhist.learning.model import DistributionalWhistMLP
from alphawhist.learning.neural_rollout import (
    _Branch, _advance, evaluate_neural_self,
)
from alphawhist.learning.dataset import read_manifest, split_files, load_shard
from benchmarks.export_training_labels import label_traces
from benchmarks.generate_training_games import generate


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(12)
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    path = tmp_path / "frozen.pt"
    save_checkpoint(path, model, metadata={"teacher_policy": "bid_aware_2"})
    return path


def snapshot():
    state = InformationState.new_round(
        ("14h", "2s", "3c"), num_players=3, round_index=0,
        dealer=2, leader=0, cards_per_player=3, observer_id=0,
    )
    state = state.apply_public_event(
        {"type": "trump", "player": 2, "suit": "h"})
    for player, bid in ((0, 1), (1, 1), (2, 0)):
        state = state.apply_public_event(
            {"type": "bid", "player": player, "bid": bid})
    return state


def test_batched_neural_self_rollouts_deterministic_and_have_valid_histograms(checkpoint):
    state = snapshot()
    before = state.to_json()
    opts = dict(checkpoint=checkpoint, n_rollouts=11, seed=38271,
                opponent_policy="bid_aware_2", collect_trick_histograms=True)
    one = evaluate_neural_self(state, worlds_per_batch=2, batch_size=3, **opts)
    two = evaluate_neural_self(state, worlds_per_batch=5, batch_size=7, **opts)
    assert one == two
    assert state.to_json() == before
    assert len(one.actions) == 3
    for i, histogram in enumerate(one.additional_trick_counts):
        assert sum(histogram) == 11
        expected = sum(
            n * (1 + k if k != 1 else 1 + k + 10)
            for k, n in enumerate(histogram)
        ) / 11
        assert abs(one.values[i] - expected) < 1e-10


def test_simulation_never_puts_sampled_hands_in_observer_features(checkpoint):
    state = snapshot()
    # The real sampled opponents' hands differ between the two futures.
    # Public actions are identical, so the model MUST see identical features.
    worlds = [
        (state.own_hand, ("2h", "4d", "9c"), ("5h", "7d", "10s")),
        (state.own_hand, ("2h", "5d", "11c"), ("8h", "6d", "12s")),
    ]
    observations = []
    for world in worlds:
        branch = _Branch(
            SimState.from_snapshot(state, world), state,
            random.Random(11), 0,
        )
        _advance(branch, CARD_ID["14h"])
        _advance(branch, CARD_ID["2h"])
        observations.append(branch.observation)
        assert branch.observation.own_hand == ("2s", "3c")
        assert branch.observation.opponents[0].remaining_count == 2
    assert observations[0].to_json() == observations[1].to_json()
    assert observations[0].opponents[0].possible_cards == (
        observations[1].opponents[0].possible_cards)


def test_label_export_accepts_neural_self_teacher_and_records_provenance(
        checkpoint, tmp_path):
    traces = tmp_path / "traces"
    with contextlib.redirect_stdout(io.StringIO()):
        generate(games=2, seed=20262190, players=3, agents=("random",),
                 workers=1, output_dir=traces)
    labels = tmp_path / "neural-labels"
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = label_traces(
            sorted(traces.glob("*.jsonl")), labels, rollouts=4,
            workers=1, seed=20262190, policy="self_neural",
            max_states=5, shard_size=3, self_checkpoint=checkpoint,
        )
    assert manifest["total_states"] == 5
    assert manifest["teacher_policy"] == "self_neural"
    assert manifest["teacher_provenance"]["opponents"] == "bid_aware_2"
    assert len(manifest["teacher_provenance"]["checkpoint_sha256"]) == 64
    for split in ("train", "val", "test"):
        for path in split_files(labels, read_manifest(labels), split):
            arrays = load_shard(path)
            assert np.all(arrays["n_rollouts"] == 4)
            assert np.all(arrays["trick_counts"].sum(axis=-1)[
                arrays["legal_mask"]] == 4)
    with pytest.raises(ValueError, match="required"):
        label_traces(sorted(traces.glob("*.jsonl")),
                     tmp_path / "without-checkpoint",
                     policy="self_neural")
