"""Bid-aware two-card rollout rules, information boundaries and reproducibility."""
import random

import pytest

from alphawhist.core.EndgamePolicy import (
    _current_win_probability, _two_trick_value, two_card_values,
    pick_bid_aware_two_card,
)
from alphawhist.core.FastRollout import (
    CARD_ID, CARDS, SimState, evaluate_actions, finish_round,
    hand_mask, legal_mask,
)
from alphawhist.core.InformationState import InformationState
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer


def two_card_snapshot(n):
    state = InformationState.new_round(
        ("14h", "2c"), n, round_index=0, dealer=n - 1,
        leader=0, cards_per_player=2
    )
    state = state.apply_public_event({"type": "trump", "player": n - 1, "suit": "h"})
    for pid in range(n):
        state = state.apply_public_event(
            {"type": "bid", "player": pid, "bid": 1}
        )
    return state


@pytest.mark.parametrize("needed,expected", [
    (0, 2.0), (1, 2.0), (2, 12.0), (-1, 2.0), (3, 2.0),
])
def test_two_trick_value_for_guaranteed_two_wins(needed, expected):
    assert _two_trick_value(1.0, 1.0, 0.0, needed) == expected


@pytest.mark.parametrize("needed,expected", [
    (0, 10.0), (1, 0.0), (2, 0.0)
])
def test_two_trick_value_for_guaranteed_losses(needed, expected):
    assert _two_trick_value(0.0, 0.0, 0.0, needed) == expected


def test_cannot_win_current_trick_if_already_beaten():
    # Our two cards are both below the already-played ace of trump.
    state = SimState(
        hands=[hand_mask(("14h", "7d")), hand_mask(("2h", "3h")),
               hand_mask(("4s", "8c"))],
        pile=[CARD_ID["14h"]],
        bids=(0, 1, 1), tricks_won=[0, 0, 0],
        leader=0, turn=1, trump=0
    )
    legal = legal_mask(state.hands[1], state.pile)
    assert _current_win_probability(state, CARD_ID["2h"], CARD_ID["3h"]) == 0.0
    assert _current_win_probability(state, CARD_ID["3h"], CARD_ID["2h"]) == 0.0
    assert set((two_card_values(state, legal) or ())[:2]) == {
        CARD_ID["2h"], CARD_ID["3h"]
    }


def test_public_state_only_even_if_other_hands_change():
    """Same acting hand and public data must give the same action distribution."""
    own = hand_mask(("14h", "2c"))
    s1 = SimState(
        hands=[own, hand_mask(("3h", "4h")), hand_mask(("5h", "6h")),
               hand_mask(("7h", "8h"))],
        pile=[], bids=(1, 1, 1, 1), tricks_won=[0]*4,
        leader=0, turn=0, trump=0
    )
    s2 = SimState(
        hands=[own, hand_mask(("3c", "4c")), hand_mask(("5d", "6d")),
               hand_mask(("7s", "8s"))],
        pile=[], bids=(1, 1, 1, 1), tricks_won=[0]*4,
        leader=0, turn=0, trump=0
    )
    allowed = legal_mask(own, ())
    assert two_card_values(s1, allowed) == two_card_values(s2, allowed)
    assert [pick_bid_aware_two_card(s1, allowed, random.Random(i))
            for i in range(30)] == [
                pick_bid_aware_two_card(s2, allowed, random.Random(i))
                for i in range(30)
            ]


def test_follow_suit_forces_one_card():
    state = SimState(
        hands=[hand_mask(("9h", "4c")), hand_mask(("10d", "8s"))],
        pile=[CARD_ID["2h"]], bids=(1, 1), tricks_won=[0, 0],
        leader=1, turn=0, trump=3
    )
    allowed = legal_mask(state.hands[0], state.pile)
    assert allowed.bit_count() == 1
    assert two_card_values(state, allowed) is None
    assert pick_bid_aware_two_card(state, allowed, random.Random(9)) == CARD_ID["9h"]


@pytest.mark.parametrize("n", range(2, 8))
def test_two_card_policy_finishes_legal_rounds_for_two_to_seven_players(n):
    for seed in range(12):
        rng = random.Random(1000 + 100 * n + seed)
        ids = rng.sample(range(52), 2*n)
        hands = [sum(1 << c for c in ids[2*i:2*i + 2]) for i in range(n)]
        state = SimState(
            hands=hands, pile=[], bids=tuple(rng.randrange(3) for _ in range(n)),
            tricks_won=[0] * n, leader=0, turn=0,
            trump=rng.randrange(4)
        )
        scores = finish_round(state, rng, policy="bid_aware_2")
        assert sum(state.tricks_won) == 2
        assert all(not mask for mask in state.hands)
        assert not state.pile
        assert len(scores) == n
        assert all(0 <= val <= 12 for val in scores)


@pytest.mark.parametrize("n", range(2, 8))
def test_seeded_action_evaluation_is_identical_across_workers(n):
    state = two_card_snapshot(n)
    reference = evaluate_actions(
        state, n_rollouts=16, seed=700+n,
        policy="bid_aware_2", workers=1, batch_size=16,
    )
    observed = evaluate_actions(
        state, n_rollouts=16, seed=700+n,
        policy="bid_aware_2", workers=2, batch_size=5,
    )
    assert observed == reference
    assert len(observed.mean_round_scores) == 2
    assert all(len(row) == n for row in observed.mean_round_scores)


def test_old_random_policy_is_unchanged_and_opt_in_differs():
    state = two_card_snapshot(3)
    original = evaluate_actions(state, n_rollouts=40, seed=400)
    random_explicit = evaluate_actions(
        state, n_rollouts=40, seed=400, policy="random"
    )
    bid_aware = evaluate_actions(
        state, n_rollouts=40, seed=400, policy="bid_aware_2"
    )
    assert original == random_explicit
    assert bid_aware.rollouts == original.rollouts
    assert bid_aware.actions == original.actions
    assert bid_aware != original


@pytest.mark.parametrize("cls", [MonteCarloPlayer, SnakeMonteCarloPlayer])
def test_agents_allow_opt_in_and_reject_bad_config(cls):
    agent = cls(0, ["14h", "2c"])
    assert agent.monteCarloPolicy == "random"
    opted = cls(0, ["14h", "2c"], additionalParameters={
        "monteCarloPolicy": "bid_aware_2", "monteCarloNumber": "16"
    })
    assert opted.monteCarloPolicy == "bid_aware_2"
    with pytest.raises(ValueError, match="policy"):
        cls(0, ["14h", "2c"], additionalParameters={
            "monteCarloPolicy": "cheat"
        })
