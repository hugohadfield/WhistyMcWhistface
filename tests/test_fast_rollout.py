"""Fast Monte Carlo regression and rule-equivalence tests."""
import random

import pytest

from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistLib import analyse_pile, all_cards_in_deck, tricksToScore
from alphawhist.core.FastRollout import (
    CARD_ID, CARDS, Evaluation, SimState, card_ids, evaluate_actions,
    finish_round, hand_mask, legal_mask, trick_winner,
)


def snapshot(hand=("14h", "2s", "3c"), n=3, bids=(1, 1, 0)):
    s = InformationState.new_round(
        hand, n, round_index=0, dealer=n - 1,
        leader=0, cards_per_player=len(hand)
    )
    s = s.apply_public_event({"type": "trump", "player": n - 1, "suit": "h"})
    for pid, bid in enumerate(bids):
        s = s.apply_public_event({"type": "bid", "player": pid, "bid": bid})
    return s


@pytest.mark.parametrize("hand", [
    [], ["14h"], ["2s", "5c", "11h"], all_cards_in_deck
])
def test_hand_masks_roundtrip(hand):
    assert [CARDS[i] for i in card_ids(hand_mask(hand))] == sorted(
        hand, key=CARD_ID.__getitem__
    )


def test_duplicate_cards_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        hand_mask(["2h", "2h"])


@pytest.mark.parametrize("seed", range(50))
def test_fast_trick_winner_matches_reference(seed):
    rng = random.Random(seed)
    for _ in range(30):
        pile = rng.sample(all_cards_in_deck, 3)
        trump = rng.choice("hcsd")
        assert trick_winner([CARD_ID[c] for c in pile], "hcsd".index(trump)) == (
            analyse_pile(pile, trump)
        )


def test_legal_mask_follows_suit():
    hand = hand_mask(["14h", "2h", "3s"])
    leading = [CARD_ID["10h"]]
    legal = legal_mask(hand, leading)
    assert set(card_ids(legal)) == {CARD_ID["14h"], CARD_ID["2h"]}


def test_legal_mask_allows_discard_when_void():
    hand = hand_mask(["14h", "3s"])
    assert legal_mask(hand, [CARD_ID["4d"]]) == hand


def test_illegal_card_rejected_without_state_mutation():
    s = snapshot()
    hands = [s.own_hand, ("2h", "4s", "5s"), ("3h", "6h", "7d")]
    sim = SimState.from_snapshot(s, hands)
    with pytest.raises(ValueError, match="not held"):
        sim.play(CARD_ID["9s"])
    assert sim.hands[0] == hand_mask(s.own_hand)
    assert sim.turn == s.turn


def test_illegal_following_suit_rejected():
    s = snapshot()
    hands = [s.own_hand, ("2h", "4s", "5s"), ("3h", "6h", "7d")]
    sim = SimState.from_snapshot(s, hands)
    sim.play(CARD_ID["14h"])
    with pytest.raises(ValueError, match="follow suit"):
        sim.play(CARD_ID["4s"])


def test_fast_tricks_advance_leader_correctly():
    s = snapshot(hand=("14h", "2s"), bids=(1, 0, 0))
    hands = [s.own_hand, ("2h", "4s"), ("3h", "6d")]
    sim = SimState.from_snapshot(s, hands)
    for card in ("14h", "2h", "3h"):
        sim.play(CARD_ID[card])
    assert sim.turn == sim.leader == 0
    assert sim.tricks_won == [1, 0, 0]
    assert sim.pile == []
    for card in ("2s", "4s", "6d"):
        sim.play(CARD_ID[card])
    assert sim.tricks_won == [1, 1, 0]
    assert sim.round_scores() == (11, 1, 10)


def test_unfinished_trick_cannot_be_scored():
    sim = SimState.from_snapshot(
        snapshot(hand=("14h",)), [("14h",), ("2h",), ("3h",)]
    )
    with pytest.raises(ValueError, match="not finished"):
        sim.round_scores()


@pytest.mark.parametrize("seed", range(40))
def test_simulated_random_play_obeys_whist_rules_and_score(seed):
    rng = random.Random(seed)
    own = tuple(rng.sample(all_cards_in_deck, 4))
    s = snapshot(own, bids=(1, 1, 0))
    hands = s.sample_hands(random.Random(seed + 10000))
    sim = SimState.from_snapshot(s, hands)
    # A second, independently implemented reference pile evaluator is used.
    expected_wins = [0, 0, 0]
    actual_hands = [list(h) for h in hands]
    leader = 0
    for _ in range(4):
        pile = []
        for pos in range(3):
            pid = (leader + pos) % 3
            hand = actual_hands[pid]
            if pile:
                suited = [card for card in hand if card[-1] == pile[0][-1]]
                legal = suited or hand
            else:
                legal = hand
            chosen = rng.choice(legal)
            hand.remove(chosen)
            pile.append(chosen)
            sim.play(CARD_ID[chosen])
        winner = (leader + analyse_pile(pile, "h")) % 3
        expected_wins[winner] += 1
        leader = winner
    assert sim.tricks_won == expected_wins
    assert sim.round_scores() == tuple(tricksToScore(b, w)
                                       for b, w in zip(s.bids, expected_wins))


def test_finish_round_from_mid_trick():
    s = snapshot(hand=("14h", "2s"), bids=(1, 0, 0))
    s = s.apply_public_event({"type": "card", "player": 0, "card": "14h"})
    s = s.apply_public_event({"type": "card", "player": 1, "card": "3s"})
    hands = [s.own_hand, ("2h",), ("3h", "6d")]
    sim = SimState.from_snapshot(s, hands)
    out = finish_round(sim, random.Random(7))
    assert sim.pile == []
    assert sum(sim.tricks_won) == 2
    assert len(out) == 3


def test_exact_ace_of_trumps_score_and_action_value():
    s = snapshot(hand=("14h",), bids=(1, 0, 0))
    res = evaluate_actions(s, n_rollouts=50, seed=123)
    assert res.actions == ("14h",)
    assert res.mean_round_scores[0][0] == 11
    assert res.values == (11.0,)


def test_candidate_deals_and_snapshots_are_not_modified():
    s = snapshot()
    before = s.to_json()
    a = evaluate_actions(s, ["14h", "2s"], n_rollouts=35, seed=3)
    b = evaluate_actions(s, ["14h", "2s"], n_rollouts=35, seed=3)
    assert a == b
    assert s.to_json() == before


@pytest.mark.parametrize("workers,batch_size", [(1, 1), (1, 13), (2, 1), (2, 19)])
def test_worker_and_batch_count_do_not_change_results(workers, batch_size):
    s = snapshot()
    ref = evaluate_actions(s, ("14h", "2s", "3c"),
                           n_rollouts=53, seed=891, workers=1, batch_size=256)
    observed = evaluate_actions(s, ("14h", "2s", "3c"),
                                n_rollouts=53, seed=891, workers=workers,
                                batch_size=batch_size)
    assert observed == ref


def test_margin_objective_preserves_expected_score_vectors():
    s = snapshot()
    a = evaluate_actions(s, n_rollouts=30, seed=11, objective="score")
    b = evaluate_actions(s, n_rollouts=30, seed=11, objective="margin")
    assert a.mean_round_scores == b.mean_round_scores
    assert b.values == tuple(
        row[0] - max(row[1:]) for row in b.mean_round_scores
    )


def test_alternative_high_card_policy_and_custom_policy():
    s = snapshot()
    high = evaluate_actions(s, n_rollouts=5, seed=10, policy="highest")
    assert isinstance(high, Evaluation)
    def choose_lowest(state, legal, rng):
        return next(card_ids(legal))
    low = evaluate_actions(s, n_rollouts=5, seed=10, policy=choose_lowest)
    assert len(low.values) == len(high.values)


@pytest.mark.parametrize("kwargs", [
    {"n_rollouts": 0}, {"workers": 0}, {"batch_size": 0},
    {"objective": "oops"}, {"policy": "illegal"},
])
def test_bad_evaluation_options_raise(kwargs):
    with pytest.raises(ValueError):
        evaluate_actions(snapshot(), n_rollouts=2, **{
            k: v for k, v in kwargs.items()
        })


def test_illegal_candidate_rejected():
    with pytest.raises(ValueError, match="legal"):
        evaluate_actions(snapshot(), ["4h"], n_rollouts=5)


def test_only_active_observer_can_evaluate():
    s = snapshot()
    s = s.apply_public_event({"type": "card", "player": 0, "card": "14h"})
    with pytest.raises(ValueError, match="observer"):
        evaluate_actions(s, n_rollouts=10)


def test_evaluate_many_worlds_and_moves_without_history_mutation():
    s = snapshot()
    before = len(s.public_history)
    result = evaluate_actions(s, n_rollouts=300, seed=2026)
    assert len(result.mean_round_scores) == 3
    assert all(len(row) == 3 for row in result.mean_round_scores)
    assert len(s.public_history) == before
