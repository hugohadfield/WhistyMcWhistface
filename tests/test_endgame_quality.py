"""Real-game two-card quality benchmark regression and independence checks."""
import random

import pytest

from benchmarks.endgame_quality import (
    _summary, collect_state, heuristics, paired_reference, run
)
from alphawhist.core.FastRollout import CARD_ID, SimState, legal_mask, hand_mask


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize("round_size", (2, 4, 7))
def test_capture_legitimate_endgame_from_completed_game(n, round_size):
    state = collect_state(n, 20261101 + n * 101 + round_size, round_size)
    assert state.num_players == n
    assert state.phase == "playing"
    assert state.turn == state.observer_id
    assert len(state.own_hand) == 2
    assert state.cards_per_player == round_size
    assert state.trick_number == round_size - 2
    assert sum(state.tricks_won) == round_size - 2
    assert sum(1 for e in state.public_history if e["type"] == "bid") == n
    assert sum(1 for e in state.public_history if e["type"] == "trump") == 1
    assert len(state.public_history) > n + 1
    assert state.pile == () or all(pid != state.observer_id for pid, _ in state.pile)
    a, b, *_ = heuristics(state)
    assert a != b and {a,b} == set(
        CARD_ID[c] for c in state.own_hand
    )


@pytest.mark.parametrize("n", range(2, 8))
def test_paired_reference_is_reproducible_and_finite(n):
    state = collect_state(n, 25001 + n, 5)
    a, b, *_ = heuristics(state)
    ref = paired_reference(state, a, b, rollouts=20, seed=5181)
    assert ref == paired_reference(state, a, b, rollouts=20, seed=5181)
    assert 0 <= ref[0] <= 12 and 0 <= ref[1] <= 12
    assert ref[2] >= 0


@pytest.mark.parametrize("n", (2, 3, 7))
def test_small_quality_roundtrip(n, tmp_path):
    out, stats = run(n, states=3, rollouts=16, seed=933,
                     csv_path=tmp_path / "quality.csv", progress=3)
    assert len(out) == 3
    assert stats["states"] == 3
    assert sum(x["players"] == n for x in out) == 3
    assert all(x["regret_random"] >= 0 for x in out)
    assert all(x["regret_softmax"] >= -1e-12 for x in out)
    assert all(x["regret_highlow"] >= -1e-12 for x in out)
    assert len((tmp_path / "quality.csv").read_text().splitlines()) == 4


def test_regret_baseline_identities():
    rows = [
        {"regret_bid_aware": 0.0, "regret_softmax": 1.0,
         "regret_highlow": 2.0, "regret_random": 2.0,
         "score_gain_softmax_vs_random": 1.0,
         "score_gain_greedy_vs_random": 2.0,
         "score_gain_highlow_vs_random": 0.0,
         "agrees": True, "clear_gap": False, "gap": 4.0},
        {"regret_bid_aware": 4.0, "regret_softmax": 3.0,
         "regret_highlow": 0.0, "regret_random": 2.0,
         "score_gain_softmax_vs_random": -1.0,
         "score_gain_greedy_vs_random": -2.0,
         "score_gain_highlow_vs_random": 2.0,
         "agrees": False, "clear_gap": True, "gap": 4.0}
    ]
    summary = _summary(rows)
    assert summary["states"] == 2
    assert summary["agreement"] == 0.5
    assert summary["clear_states"] == 1
    assert summary["clear_agreement"] == 0.0
    assert summary["score_gain_softmax_vs_random"] == 0
    assert summary["regret_random"] == 2
