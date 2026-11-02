"""Safety, legality, reproducibility and real-game checks for MC bidding."""
import contextlib
import io
import json
from dataclasses import replace

import pytest

from alphawhist.core.InformationState import InformationState
from alphawhist.core.BidRollout import (
    evaluate_bids, legal_future_bids, public_zero_streaks, _approx_future_bids,
)
from alphawhist.core.WhistStructure import Game
from MonteCarloBidPlayer import MonteCarloBidPlayer
from MonteCarloPlayer import MonteCarloPlayer
from benchmarks.mc_bid_tournament import tournament


def bidding_snapshot(*, dealer=0, observer=1, cards=3, hand=None):
    if hand is None:
        hand = ["14h", "8c", "3d"][:cards]
    state = InformationState.new_round(
        own_hand=hand, num_players=3, round_index=0,
        dealer=dealer, leader=(dealer+1)%3, cards_per_player=cards,
        observer_id=observer,
    )
    return state.apply_public_event(
        {"type": "trump", "player": dealer, "suit": "s"})


def test_evaluates_all_legal_bids_and_score_formula():
    state = bidding_snapshot()
    estimate = evaluate_bids(state, range(4), n_rollouts=96, seed=77)
    assert estimate.bids == (0, 1, 2, 3)
    assert all(0 <= x <= 1 for x in estimate.exact_bid_probabilities)
    assert all(0 <= x <= state.cards_per_player for x in estimate.mean_tricks)
    for mean, p, expected in zip(
        estimate.mean_tricks, estimate.exact_bid_probabilities,
        estimate.expected_round_scores,
    ):
        assert mean + 10*p == pytest.approx(expected)
    assert estimate.best_bid() in range(4)
    assert estimate == evaluate_bids(state, range(4), n_rollouts=96, seed=77)


def test_dealer_forbidden_sum_and_known_previous_bids():
    state = bidding_snapshot(observer=1, dealer=0)
    state = state.apply_public_event({"type": "bid", "player": 1, "bid": 1})
    state = state.apply_public_event({"type": "bid", "player": 2, "bid": 1})
    # At this stage the dealer knows two real bids.
    dealer = InformationState.new_round(
        own_hand=["14h","8c","3d"], num_players=3,
        round_index=0, dealer=0, leader=1, cards_per_player=3, observer_id=0)
    dealer = dealer.apply_public_event(
        {"type": "trump", "player": 0, "suit": "s"})
    dealer = dealer.apply_public_event({"type":"bid","player":1,"bid":1})
    dealer = dealer.apply_public_event({"type":"bid","player":2,"bid":1})
    assert dealer.turn == 0
    assert legal_future_bids(0, dealer.bids, public_zero_streaks(dealer), 3, 0) == (0,2,3)
    with pytest.raises(ValueError, match="Illegal"):
        evaluate_bids(dealer, (1,), n_rollouts=8)
    good = evaluate_bids(dealer, (0,2,3), n_rollouts=32, seed=321)
    assert good.best_bid() in (0,2,3)


def test_model_future_dealer_cannot_make_total_equal_number_of_tricks():
    state = bidding_snapshot(observer=1, dealer=0)
    guessed = _approx_future_bids(state, 1, public_zero_streaks(state))
    assert guessed[1] == 1
    assert sum(guessed) != 3
    assert all(isinstance(x, int) for x in guessed)


def test_observation_contains_no_true_hidden_hands():
    # This snapshot defines all information the agent may use; no Game,
    # opponent real hands or privileged replay file is available.
    state = bidding_snapshot()
    first = evaluate_bids(state, (0,1,2,3), n_rollouts=24, seed=6)
    clone = InformationState.from_dict(state.to_dict())
    assert evaluate_bids(clone, (0,1,2,3), n_rollouts=24, seed=6) == first


def test_zero_streak_from_public_bids_is_respected():
    previous = tuple(
        json.dumps({"type":"bid","player":1,"bid":0})
        for _ in range(2))
    state = InformationState.new_round(
        own_hand=["14h","8c"], num_players=3, round_index=2,
        dealer=0, leader=1, cards_per_player=2,
        observer_id=1, previous_history=previous)
    state = state.apply_public_event(
        {"type":"trump","player":0,"suit":"d"})
    assert public_zero_streaks(state)[1] == 2
    with pytest.raises(ValueError, match="Illegal"):
        evaluate_bids(state, (0,), n_rollouts=8)
    assert evaluate_bids(state, (1,2), n_rollouts=8).best_bid() in (1,2)


def test_invalid_snapshots_and_budgets():
    state = bidding_snapshot()
    with pytest.raises(ValueError, match="n_rollouts"):
        evaluate_bids(state, (1,), n_rollouts=0)
    with pytest.raises(ValueError, match="distinct"):
        evaluate_bids(state, (1,1), n_rollouts=3)
    with pytest.raises(ValueError, match="Illegal"):
        evaluate_bids(state, (4,), n_rollouts=3)
    with pytest.raises(ValueError, match="Expected snapshot"):
        evaluate_bids(
            state.apply_public_event({"type":"bid","player":1,"bid":1}),
            (1,), n_rollouts=3)
    with pytest.raises(ValueError, match="positive"):
        MonteCarloBidPlayer(0, additionalParameters={"bidRollouts": "0"})


def test_real_whist_engine_uses_mc_bid_hook_and_same_card_search():
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [3, 2]
    mc_bidder = MonteCarloBidPlayer(1, additionalParameters={
        "monteCarloNumber": "8", "monteCarloPolicy": "bid_aware_2",
        "bidRollouts": "24"})
    game.players[1] = mc_bidder
    assert isinstance(mc_bidder, MonteCarloPlayer)
    assert mc_bidder.monteCarloNumber == 8
    assert mc_bidder.monteCarloPolicy == "bid_aware_2"
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber == 2
    assert sum(game.tricksWon) == 2
    assert mc_bidder.bid in range(3)


def test_seat_swapped_tournament_real_games(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        results, summary = tournament(
            games=2, seed=9037, play_rollouts=4, bid_rollouts=12,
            workers=1, bootstrap=100, progress_every=0,
            json_path=tmp_path/"tournament.json",
            csv_path=tmp_path/"tournament.csv")
    assert len(results) == 2
    assert results[0].deal_seed == results[1].deal_seed
    assert results[0].challenger_seat == results[1].baseline_seat
    assert results[1].challenger_seat == results[0].baseline_seat
    assert summary["paired_deals"] == 1
    assert summary["card_play_rollouts"] == 4
    assert sum([summary["challenger_head_to_head_wins"],
                summary["baseline_head_to_head_wins"],
                summary["head_to_head_ties"]]) == 2
    assert (tmp_path/"tournament.json").exists()
    assert (tmp_path/"tournament.csv").exists()



def test_forced_zero_dealer_when_otherwise_no_legal_bid():
    # If the dealer has two consecutive zero bids and the other players
    # bid zero in a one-card round, dealer=0 is the only possible legal
    # action that still preserves the sum constraint.
    history = tuple(json.dumps({"type": "bid", "player": 0, "bid": 0})
                    for _ in range(2))
    state = InformationState.new_round(
        own_hand=["14h"], num_players=3, round_index=2,
        dealer=0, leader=1, cards_per_player=1,
        observer_id=0, previous_history=history)
    state = state.apply_public_event(
        {"type": "trump", "player": 0, "suit": "s"})
    state = state.apply_public_event({"type":"bid","player":1,"bid":0})
    state = state.apply_public_event({"type":"bid","player":2,"bid":0})
    assert public_zero_streaks(state)[0] == 2
    assert legal_future_bids(0, state.bids, public_zero_streaks(state), 1, 0) == (0,)
    player = MonteCarloBidPlayer(0)
    player.resetZeroCounter(2)
    assert player.getValidBidOptions(3, 1, [0, 0]) == [0]
    estimate = evaluate_bids(state, (0,), n_rollouts=16, seed=23)
    assert estimate.best_bid() == 0


def test_future_bids_preserve_forced_dealer_exception():
    history = tuple(json.dumps({"type": "bid", "player": 0, "bid": 0})
                    for _ in range(2))
    state = InformationState.new_round(
        own_hand=["14h"], num_players=3, round_index=2,
        dealer=0, leader=1, cards_per_player=1,
        observer_id=1, previous_history=history)
    state = state.apply_public_event(
        {"type":"trump","player":0,"suit":"s"})
    predicted = _approx_future_bids(state, 0, public_zero_streaks(state))
    assert predicted == (0, 0, 0)
    assert sum(predicted) != 1
    assert evaluate_bids(state, (0, 1), n_rollouts=16).best_bid() in (0, 1)
