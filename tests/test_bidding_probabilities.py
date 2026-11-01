"""Tests for legal trick-win probabilities and lead transitions."""
import random
import pytest
from alphawhist.core import WhistLib as w
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer
from RandomWithAdvancedBidPlayer import RandomWithAdvancedBidPlayer
from StatHeuristicPlayer import StatHeuristicPlayer


def test_ace_of_trumps_guaranteed_for_leading_and_following():
    cards = ['14h', '2s', '3s', '4c', '5c', '6d', '7d']
    lead, follow = w.estimate_card_win_probabilities(cards, 3, 7, 'h', 200)
    assert lead[0] == follow[0] == 1.0


def test_one_card_probabilities_match_exact_combinatorics():
    random.seed(13)
    lead, follow = w.estimate_card_win_probabilities(['14s'], 2, 1, 'h', 5000)
    assert lead[0] == pytest.approx(38 / 51, abs=.03)
    assert follow[0] == pytest.approx(12 / 51, abs=.03)


@pytest.mark.parametrize('cls,params', [
    (MonteCarloPlayer, {'monteCarloNumber': '2'}),
    (SnakeMonteCarloPlayer, {'monteCarloNumber': '2'}),
    (RandomWithAdvancedBidPlayer, None),
])
def test_all_bidder_wrappers_exclude_own_cards(cls, params):
    player = cls(0, ['14h', '2s', '3s'], additionalParameters=params)
    assert player.computeLeadingCardVictory(3, player.real_hand, 3, 'h')[0] == 1
    assert player.computeFollowingCardVictory(3, player.real_hand, 3, 'h')[0] == 1


def test_heuristic_uses_new_probability_model():
    player = StatHeuristicPlayer(0, ['14h', '3s'])
    assert player.computeExpectedRewardForLedCard('14h', 2, player.real_hand, 2, 'h', 1, [])[0] == 1


def test_lead_changes_after_losing(monkeypatch):
    monkeypatch.setattr(w.random, 'randrange', lambda n: 0)
    assert w.mini_monte_sim([0, 0], [1, 1], 0, 20) == [0, 1, 0]


def test_following_win_makes_next_trick_lead(monkeypatch):
    monkeypatch.setattr(w.random, 'randrange', lambda n: 0)
    assert w.mini_monte_sim([1, 1], [1, 0], 1, 20) == [0, 0, 1]


def test_inputs_unchanged():
    lead, follow = [0.4, .8], [.2, .6]
    w.mini_monte_sim(lead, follow, 0, 10)
    assert lead == [0.4, .8] and follow == [.2, .6]


def test_estimator_impossible_deal_fails():
    with pytest.raises(ValueError, match='Not enough unseen cards'):
        w.estimate_card_win_probabilities(['2h'], 5, 13, 'h')


def test_empty_hand_still_supported():
    assert w.estimate_card_win_probabilities([], 3, 0, 'h') == ([], [])


@pytest.mark.parametrize('cls,params', [
    (MonteCarloPlayer, {'monteCarloNumber': '2'}),
    (SnakeMonteCarloPlayer, {'monteCarloNumber': '2'}),
    (RandomWithAdvancedBidPlayer, None),
    (StatHeuristicPlayer, None),
])
def test_final_bidder_must_not_make_total_equal_tricks(cls, params):
    # An ace of trumps is a certain winner, so the estimator strongly
    # prefers bidding 1. But 1 is illegal when the earlier bids sum to 0
    # in a one-trick round.
    player = cls(2, ['14h'], additionalParameters=params)
    bid = player.makeBid(3, 1, 'h', [0, 0])
    assert bid == 0
    assert player.zero_bid_count == 1
