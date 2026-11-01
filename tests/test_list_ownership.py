"""Regression tests for Python list ownership and consistent Whist state."""
import random
import pytest

from alphawhist.core import WhistLib as w
from alphawhist.core.WhistPlayer import Player
from alphawhist.core.WhistStructure import Game
from RandomPlayer import RandomPlayer


def game_with_hands(*hands):
    game = Game.__new__(Game)
    game.numberofplayers = len(hands)
    game.players = [Player(i, hand) for i, hand in enumerate(hands)]
    game.cardnumbers = [len(hands[0])]
    game.resetGame()
    return game


def test_constructor_does_not_alias_hands():
    cards = ['2h', '3s']
    p = Player(0, cards)
    cards.clear()
    assert p.real_hand == p.possible_hand == ['2h', '3s']


@pytest.mark.parametrize('name', ['real_hand', 'possible_hand'])
def test_property_getters_return_copies(name):
    p = Player(0, ['2h', '3s'])
    returned = getattr(p, name)
    returned.clear()
    assert getattr(p, name) == ['2h', '3s']


@pytest.mark.parametrize('method,pile', [
    ('getPossibleMoves', []), ('getRealPossibleMoves', []),
    ('getPossibleMoves', ['11c']), ('getRealPossibleMoves', ['11c']),
    ('getPossibleMoves', ['14s']), ('getRealPossibleMoves', ['14s']),
])
def test_move_lists_do_not_alias_hand(method, pile):
    p = Player(0, ['2h', '3s'])
    getattr(p, method)(pile).clear()
    assert p.real_hand == p.possible_hand == ['2h', '3s']


def test_setters_copy_input():
    p = Player(0, ['2h'])
    possible = ['2h', '3s']
    p.possible_hand = possible
    possible.clear()
    real = ['2h']
    p.real_hand = real
    real.clear()
    assert p.real_hand == ['2h']
    assert p.possible_hand == ['2h', '3s']


def test_set_hands_atomically_and_independently():
    p = Player(0, ['2h'])
    real = ['3s']
    possible = ['3s', '4d']
    p.set_hands(real, possible)
    real.clear()
    possible.clear()
    assert p.real_hand == ['3s'] and p.possible_hand == ['3s', '4d']


def test_invalid_atomic_assignment_preserves_old_state():
    p = Player(0, ['2h'])
    with pytest.raises(ValueError, match='contain real hand'):
        p.set_hands(['3s'], ['2h'])
    assert p.real_hand == p.possible_hand == ['2h']


def test_direct_real_assignment_requires_possible_card():
    p = Player(0, ['2h'])
    with pytest.raises(ValueError, match='contain real hand'):
        p.real_hand = ['3s']
    assert p.real_hand == ['2h']


def test_duplicate_possible_cards_rejected():
    p = Player(0, ['2h'])
    with pytest.raises(ValueError, match='duplicates'):
        p.possible_hand = ['2h', '2h']
    assert p.possible_hand == ['2h']


def test_redeal_does_not_mutate_global_deck():
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    original = w.all_cards_in_deck.copy()
    random.seed(192)
    g.randomDeal(5)
    hands = [p.real_hand for p in g.players]
    assert len(set(card for hand in hands for card in hand)) == 15
    g.players[1].removeReal(g.players[1].real_hand[0])
    assert w.all_cards_in_deck == original


def test_determinization_preserves_known_hand_and_unique_cards():
    g = game_with_hands(['2h', '3h'], ['4h', '5h'], ['6h', '7h'])
    for p in g.players[1:]:
        p.possible_hand = ['4h', '5h', '6h', '7h']
    random.seed(206)
    g.convertToDeterministicGame()
    assert g.players[0].real_hand == ['2h', '3h']
    hands = [p.real_hand for p in g.players]
    assert len(set(c for h in hands for c in h)) == 6
    assert all(p.real_hand == p.possible_hand for p in g.players)


def test_determinization_respects_constraints():
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    g.players[1].possible_hand = ['3h']
    g.players[2].possible_hand = ['3h', '4h']
    g.convertToDeterministicGame()
    assert g.players[1].real_hand == ['3h']
    assert g.players[2].real_hand == ['4h']


def test_failed_determinization_does_not_corrupt_state():
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    g.players[1].possible_hand = ['3h']
    g.players[2].set_hands(['3h'])
    before = [(p.real_hand, p.possible_hand) for p in g.players]
    with pytest.raises(ValueError, match='consistent'):
        g.convertToDeterministicGame()
    assert [(p.real_hand, p.possible_hand) for p in g.players] == before


def test_bid_callback_cannot_corrupt_bid_history():
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    g.resetRound()
    seen = []
    class ModifyingBidder(Player):
        def makeBid(self, nplayers, ncards, trump, previousbids):
            seen.append(tuple(previousbids))
            previousbids.append(99)
            return 0
    g.players = [ModifyingBidder(i) for i in range(3)]
    g.getPlayerBids()
    assert seen == [(), (0,), (0, 0)]
    assert g.bids == [0, 0, 0]


def test_move_callback_cannot_corrupt_live_pile(capsys):
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    g.resetRound()
    g.leader = g.turn = 0
    class ModifyingPlayer(Player):
        def makeMove(self, pile, trump, full_game):
            pile.append('14s')
            card = self.real_hand[0]
            assert self.playCard(card)
            return card
    g.players = [ModifyingPlayer(i, [card]) for i, card in enumerate(['2h', '3h', '4h'])]
    g.playFullTrick()
    assert g.pile == ['2h', '3h', '4h']
    assert g.tricksWon == [0, 0, 1]


def test_manual_deal_gives_opponents_disjoint_actual_hands(monkeypatch):
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    monkeypatch.setattr('alphawhist.core.WhistStructure.generate_manual_deal', lambda count: ['2h', '3h'])
    g.manualDeal(2)
    hands = [p.real_hand for p in g.players]
    assert hands[0] == ['2h', '3h']
    assert all(len(h) == 2 for h in hands)
    assert len({card for hand in hands for card in hand}) == 6
    assert all('2h' not in p.possible_hand for p in g.players[1:])


def test_manual_deal_rejects_duplicates_without_state_change(monkeypatch):
    g = game_with_hands(['2h'], ['3h'], ['4h'])
    monkeypatch.setattr('alphawhist.core.WhistStructure.generate_manual_deal', lambda count: ['2h', '2h'])
    before = [(p.real_hand, p.possible_hand) for p in g.players]
    with pytest.raises(ValueError, match='unique'):
        g.manualDeal(2)
    assert [(p.real_hand, p.possible_hand) for p in g.players] == before


@pytest.mark.parametrize('module,klass,params', [
    ('MonteCarloPlayer','MonteCarloPlayer',{'monteCarloNumber':'2'}),
    ('SnakeMonteCarloPlayer','SnakeMonteCarloPlayer',{'monteCarloNumber':'2'}),
    ('ManualPlayer','ManualPlayer',None),
])
def test_model_player_preserves_hand_without_sharing(module, klass, params):
    import importlib
    cls = getattr(importlib.import_module(module), klass)
    bot = cls(0, ['2h'], additionalParameters=params)
    bot.resetZeroCounter(1)
    model = bot.generateModelPlayer()
    assert model.real_hand == ['2h']
    assert model.zero_bid_count == 1
    model.real_hand.clear()
    assert bot.real_hand == ['2h']
