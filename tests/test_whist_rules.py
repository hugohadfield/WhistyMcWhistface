"""Deterministic rules/regression tests for WhistyMcWhistface.

Run: python -m pytest -q tests/test_whist_rules.py
Regressions for gameplay rules, state integrity and formerly broken cases.
"""
import itertools
import math
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from alphawhist.core import WhistLib as w
from alphawhist.core.WhistPlayer import Player
from alphawhist.core.WhistStructure import Game
from alphawhist.core.WhistRound import Round, Trick
from alphawhist.core.TrumpPicker import picktrump, pickRandomTrump
from RandomPlayer import RandomPlayer


# --- Deck and card representations ---

def test_deck_has_52_distinct_cards():
    assert len(w.all_cards_in_deck) == len(set(w.all_cards_in_deck)) == 52


def test_deck_has_thirteen_cards_per_suit():
    assert Counter(card[-1] for card in w.all_cards_in_deck) == dict.fromkeys('hcsd', 13)


def test_deck_has_every_rank_in_every_suit():
    assert set(w.all_cards_in_deck) == {f'{rank}{suit}' for suit in 'hcsd' for rank in range(2, 15)}


def test_card_index_mapping_is_bijective():
    assert set(w.card_to_index_map.values()) == set(range(52))
    for card, index in w.card_to_index_map.items():
        assert w.index_to_card_map[index] == card


@pytest.mark.parametrize('rank,face', [(11,'J'), (12,'Q'), (13,'K'), (14,'A')])
@pytest.mark.parametrize('suit', list('hcsd'))
def test_face_card_round_trip(rank, face, suit):
    assert w.translate_card_name(f'{rank}{suit}') == f'{face}{suit}'
    assert w.translate_inv_card_name(f'{face}{suit}') == f'{rank}{suit}'


def test_number_card_round_trip():
    for rank in range(2,11):
        for suit in 'hcsd':
            c = f'{rank}{suit}'
            assert w.translate_inv_card_name(w.translate_card_name(c)) == c


@pytest.mark.parametrize('bid,taken,expected', [(0,0,10), (1,1,11), (7,7,17), (1,0,0), (0,2,2), (3,2,2)])
def test_score_matches_exact_bid_bonus(bid, taken, expected):
    assert w.tricksToScore(bid, taken) == expected


@pytest.mark.parametrize('pile,trump,winner', [
    (['2h','9h','6h'], 's', 1),
    (['9h','2s','14h'], 's', 1),
    (['9h','14s','2s'], 's', 1),
    (['9h','14d','10c'], 's', 0),
    (['10h','9h','14d'], 's', 0),
    (['13h','14h','2d'], 's', 1),
    (['14h','13h','10h'], 's', 0),
    (['2s','13h','14h'], 's', 0),
    (['2c'], 'h', 0),
    (['10s','11s','12s','13s'], 'h', 3),
    (['2h','13h','14h'], 'h', 2),
])
def test_trick_winner(pile, trump, winner):
    assert w.analyse_pile(pile, trump) == winner


def test_randomized_trick_winners_match_reference():
    random.seed(357)
    for _ in range(350):
        pile = random.sample(w.all_cards_in_deck, 4)
        trump = random.choice('hcsd')
        lead = pile[0][-1]
        winning = max(range(4), key=lambda i:(pile[i][-1] == trump,
                     pile[i][-1] == lead, int(pile[i][:-1])) if pile[i][-1] in (lead, trump)
                     else (False, False, -1))
        assert w.analyse_pile(pile, trump) == winning, (pile, trump)


# --- Deals ---

@pytest.mark.parametrize('n',[0,1,7,13,52])
def test_random_hand_card_count_and_unique(n):
    random.seed(n)
    cards = w.generate_random_hand(n)
    assert len(cards) == len(set(cards)) == n
    assert set(cards) <= set(w.all_cards_in_deck)


@pytest.mark.parametrize('n,p', [(1,2),(7,3),(13,4),(17,3),(0,4)])
def test_random_deal_is_disjoint(n,p):
    random.seed(n+p)
    hands = w.generate_random_deal(n,p)
    assert len(hands) == p
    assert all(len(h) == n for h in hands)
    cards = list(itertools.chain.from_iterable(hands))
    assert len(cards) == len(set(cards)) == n*p
    assert set(cards) <= set(w.all_cards_in_deck)


def test_deal_more_than_deck_does_not_return_hands():
    assert w.generate_random_deal(14, 4) == 0


def test_cards_of_suit_preserves_order():
    assert w.cards_of_suit(['2h','3c','11h','14s','5h'],'h') == ['2h','11h','5h']


# --- Comparisons, probability helpers, probability distributions ---

@pytest.mark.parametrize('card,trump,expected', [
    ('9h','h', {f'{i}h' for i in range(10,15)}),
    ('10h','h', {f'{i}h' for i in range(11,15)}),
    ('13h','h', {'14h'}),
    ('14h','h', set()),
    ('9h','s', {f'{i}h' for i in range(10,15)} | {f'{i}s' for i in range(2,15)}),
    ('14h','s', {f'{i}s' for i in range(2,15)}),
])
def test_card_beating_list(card,trump,expected):
    assert set(w.card_beating_list(card,trump)) == expected


@pytest.mark.parametrize('card,lead,trump,expected', [
    ('9h','h','h', {f'{i}h' for i in range(10,15)}),
    ('10h','h','h', {f'{i}h' for i in range(11,15)}),
    ('13h','h','h', {'14h'}),
    ('14h','h','h', set()),
    ('8s','h','d', {f'{i}h' for i in range(2,15)} | {f'{i}d' for i in range(2,15)}),
])
def test_card_following_list(card,lead,trump,expected):
    assert set(w.card_following_list(card,lead,trump)) == expected


def test_player_beating_list_uses_players_possible_hand():
    opponent = SimpleNamespace(possible_hand=['11h','12c','2s'])
    assert set(w.player_beating_list('10h', 's', opponent)) == {'11h','2s'}


def test_player_beating_probability():
    opponent = SimpleNamespace(possible_hand=['11h','12c','2s','3d'])
    assert w.player_beating_probability('10h', 's', opponent) == 0.5


def test_all_player_possible_cards_are_union():
    opponents = [SimpleNamespace(possible_hand=['2h','3h']),SimpleNamespace(possible_hand=['3h','4s'])]
    assert set(w.list_all_possible_player_cards(opponents)) == {'2h','3h','4s'}


def test_max_and_index_first_maximum():
    assert w.max_and_index([2,9,9,3]) == (9,1)


@pytest.mark.parametrize('probs,expected', [([0.0,0.0],[1.0,0.0,0.0]),([1.0,1.0],[0.0,0.0,1.0]),([0.0,1.0],[0.0,1.0,0.0])])
def test_monte_carlo_distribution_deterministic(probs,expected):
    assert w.monte_carlo_pdfify(probs,20) == expected


def test_monte_carlo_distribution_normalized():
    random.seed(10)
    pdf = w.monte_carlo_pdfify([0.2,0.5,0.8],1000)
    assert len(pdf)==4 and sum(pdf)==pytest.approx(1.0)
    assert all(0<=p<=1 for p in pdf)


@pytest.mark.parametrize('lead,follow,pos,expected', [
    ([1,1,1],[0,0,0],0,[0,0,0,1]),
    ([1,1,1],[0,0,0],1,[1,0,0,0]),
    ([0,0],[1,1],0,[0,1,0]),
])
def test_mini_monte_deterministic_probabilities(lead,follow,pos,expected):
    assert w.mini_monte_sim(lead,follow,pos,20) == expected


def test_mini_monte_is_normalized():
    random.seed(12)
    pdf = w.mini_monte_sim([0.2,0.3],[0.5,0.9],0,100)
    assert sum(pdf)==pytest.approx(1.0)


def test_probability_of_leading_suit():
    p=SimpleNamespace(possible_hand=['2h','3h','7s','8c'])
    assert w.probability_player_leads_suit(p,'h')==0.5


def test_empty_hand_leads_suit_with_zero_probability():
    assert w.probability_player_leads_suit(SimpleNamespace(possible_hand=[]),'h') == 0.0


# --- Player state and legal moves ---

def test_player_starts_with_specified_hand():
    p=Player(2,['3h','4s'])
    assert p.playerNumber==2 and p.n_cards_in_hand==2
    assert p.real_hand==p.possible_hand==['3h','4s']


def test_player_without_hand_has_full_possible_deck():
    assert len(Player(0).possible_hand)==52


def test_player_follows_suit_when_possible():
    p=Player(0,['2h','5s','11h'])
    assert p.getRealPossibleMoves(['13h']) == ['2h','11h']
    assert p.getPossibleMoves(['13h']) == ['2h','11h']


def test_player_can_discard_when_void():
    p=Player(0,['2h','5s'])
    assert p.getRealPossibleMoves(['13d']) == ['2h','5s']


def test_player_can_lead_any_card():
    p=Player(0,['2h','5s'])
    assert p.getRealPossibleMoves([]) == ['2h','5s']


def test_remove_possible_cards():
    p=Player(0)
    p.removePossible(['2h','3c'])
    assert '2h' not in p.possible_hand and '3c' not in p.possible_hand
    assert len(p.possible_hand)==50


def test_remove_real_cards():
    p=Player(0,['2h','3c'])
    p.removeReal('2h')
    assert p.real_hand==['3c']


def test_play_held_card_removes_from_both_hands(capsys):
    p=Player(0,['2h','3c'])
    assert p.playCard('2h') is True
    assert p.real_hand==p.possible_hand==['3c']


def test_play_unavailable_card_does_not_change_hand(capsys):
    p=Player(0,['2h','3c'])
    assert p.playCard('14h') is False
    assert p.real_hand==['2h','3c']


def test_playcard_rejects_card_not_actually_held(capsys):
    p=Player(0,['2h'])
    p.possible_hand=['2h','14s']
    assert p.playCard('14s') is False


def test_invalid_possible_hand_assignment_does_not_mutate_state(capsys):
    p=Player(0,['2h','3c'])
    with pytest.raises(ValueError):
        p.possible_hand=['2h']
    assert p.possible_hand==['2h','3c']


def test_void_suit_is_removed_from_possible_hand(capsys):
    p=Player(0)
    p.cardSuitCheck(['12h'],'2s')
    assert not w.cards_of_suit(p.possible_hand,'h')
    assert len(p.possible_hand)==39


def test_following_suit_does_not_remove_suit(capsys):
    p=Player(0)
    p.cardSuitCheck(['12h'],'2h')
    assert len(p.possible_hand)==52


@pytest.mark.parametrize('bid,count,expected', [(0,0,1),(0,1,2),(1,2,0)])
def test_zero_bid_counter(bid,count,expected,capsys):
    p=Player(0)
    p.resetZeroCounter(count)
    p.advanceZeroCounter(bid)
    assert p.zero_bid_count==expected


def test_bid_no_exact_sum_for_last_bidder():
    p=Player(0)
    assert p.checkValidBid(3,7,2,[3,2]) is False
    assert p.checkValidBid(3,7,1,[3,2]) is True


def test_no_three_zero_bids_in_a_row():
    p=Player(0)
    p.resetZeroCounter(2)
    assert p.checkValidBid(3,7,0,[]) is False


def test_bid_options_are_within_hand_size():
    p=Player(0)
    assert p.getValidBidOptions(3,3,[]) == [0,1,2,3]


@pytest.mark.parametrize('bid',[-1,8])
def test_check_valid_bid_rejects_out_of_range(bid):
    assert Player(0).checkValidBid(3,7,bid,[]) is False


def test_check_valid_bid_rejects_too_many_previous_bids():
    assert Player(0).checkValidBid(3,7,1,[1,2,3]) is False


def test_convert_to_valid_hand_stays_within_possibles(capsys):
    p=Player(0,['2h','3h'])
    p.possible_hand=['2h','3h','4h','5h']
    hand=p.convertToValidHand()
    assert len(hand)==2 and len(set(hand))==2
    assert set(hand)<=set(p.possible_hand)


@pytest.mark.parametrize('hand',[['2h'],['14s'],['2h','4h','13c'],['14h','14d','14s','14c']])
def test_trump_picker_always_returns_valid_suit(hand):
    assert picktrump(hand) in 'hcsd'


def test_random_trump_picker_returns_suit():
    assert set(pickRandomTrump() for _ in range(200)) == set('hcsd')


def test_round_tracks_basic_state():
    r=Round(3,7)
    assert (r.numberofplayers,r.numberofcards,r.trickCounter)==(3,7,0)


def test_trick_constructor_initializes_without_error():
    trick=Trick(3,0)
    assert trick.numberofplayers==3 and trick.trickCounter==0


# --- Game orchestration and scoring ---

def make_game(n=3):
    game=Game.__new__(Game)
    game.cardnumbers=[7,6,5,4,3,2,1,2,3,4,5,6,7]
    game.numberofplayers=n
    game.players=[Player(i) for i in range(n)]
    game.resetGame()
    return game


def test_game_initializes_dealer_leader_and_scores():
    g=make_game()
    assert g.dealer==0 and g.leader==1 and g.roundnumber==0
    assert g.scores==[0,0,0]


def test_game_round_reset_initializes_bids_and_tricks():
    g=make_game()
    g.resetRound()
    assert g.bids==[-1,-1,-1] and g.tricksWon==[0,0,0]


def test_game_winner_relative_to_leader():
    g=make_game(4)
    g.leader=3
    g.tricksWon=[0,0,0,0]
    g.updateTricksWon(2)
    assert g.tricksWon==[0,1,0,0]


def test_game_set_turn_to_trick_winner(capsys):
    g=make_game()
    g.leader=2
    g.pile=['2h','12s','3h']
    g.trumpsuit='s'
    g.setTurnToWinner()
    assert g.leader==g.turn==0


def test_game_end_round_rotates_dealer():
    g=make_game(3)
    g.endRound()
    assert (g.roundnumber,g.dealer,g.leader)==(1,1,2)
    g.endRound()
    assert (g.roundnumber,g.dealer,g.leader)==(2,2,0)


def test_game_score_accumulates(capsys):
    g=make_game()
    g.bids=[2,1,4]
    g.tricksWon=[2,3,2]
    g.scores=[5,0,7]
    g.processScore()
    assert g.scores==[17,3,9]


def test_game_redeal_changes_hand_without_throwing(capsys):
    g=make_game()
    random.seed(1234)
    g.randomDeal(7)
    g.randomDeal(6)
    assert all(len(p.real_hand)==6 for p in g.players)


def test_game_random_deal_produces_correct_hand_counts(capsys):
    g=make_game()
    random.seed(2345)
    g.randomDeal(7)
    assert all(len(p.real_hand)==7 for p in g.players)
    all_hands=[c for p in g.players for c in p.real_hand]
    assert len(set(all_hands))==21


def test_full_random_game_completes_all_thirteen_rounds(capsys):
    random.seed(3434)
    g=make_game()
    g.players=[RandomPlayer(i) for i in range(3)]
    # Game.playFullRound defaults to interactive manual dealing. Replace it for CI.
    g.manualDeal=g.randomDeal
    g.playFullGame()
    assert g.roundnumber==13
    assert sum(g.scores)>=0


# Further regressions identified while fixing the original failures.

def test_trump_following_another_suit_cannot_be_beaten_by_lead_suit():
    assert set(w.card_following_list('10s', 'h', 's')) == {
        f'{rank}s' for rank in range(11, 15)
    }


def test_no_possible_opponents_means_zero_beating_probability():
    assert w.player_beating_probability('10h', 's', SimpleNamespace(possible_hand=[])) == 0.0
    assert w.all_players_beating_probability('10h', 's', []) == 0.0


def test_random_player_selects_trump_from_real_hand(monkeypatch):
    p = RandomPlayer(1, ['14s'])
    p.possible_hand = ['14s', '2h', '3h']
    observed = []
    monkeypatch.setattr('RandomPlayer.TrumpPicker.picktrump', lambda hand: observed.append(list(hand)) or 's')
    assert p.pick_trumps(3) == 's'
    assert observed == [['14s']]


def test_all_bot_game_uses_random_deal_by_default(capsys):
    g = make_game(3)
    g.players = [RandomPlayer(i) for i in range(3)]
    g.cardnumbers = [2]
    def unexpected_manual_deal(_):
        raise AssertionError('Bot-only game must not ask for manual card entry')
    g.manualDeal = unexpected_manual_deal
    random.seed(5757)
    g.playFullGame()
    assert g.roundnumber == 1
    assert sum(g.tricksWon) == 2
