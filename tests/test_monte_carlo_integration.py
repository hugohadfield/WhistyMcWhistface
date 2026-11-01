"""Tests that real Monte Carlo bot decisions use the new fast evaluator."""
import random

import pytest

from alphawhist.core.WhistStructure import Game
from alphawhist.core.FastRollout import evaluate_actions
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer


def game_with_mc_player(cls, pid, ncards=3):
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [ncards]
    game.resetRound()
    random.seed(12345 + pid)
    game.randomDeal(ncards)
    cards = game.players[pid].real_hand
    game.players[pid] = cls(
        pid, cards, additionalParameters={"monteCarloNumber": "16"}
    )
    game.trumpsuit = "h"
    game._record_public_event({"type": "trump", "player": game.dealer, "suit": "h"})
    game.getPlayerBids()
    game.turn = game.leader
    return game


@pytest.mark.parametrize("cls", [MonteCarloPlayer, SnakeMonteCarloPlayer])
@pytest.mark.parametrize("pid", [0, 1, 2])
def test_mc_bot_makes_legal_moves_in_every_seat(cls, pid, capsys):
    game = game_with_mc_player(cls, pid)
    # Resolve to the MC player's turn while preserving the public history.
    while game.turn != pid:
        player = game.players[game.turn]
        card = player.makeMove(game.pile.copy(), game.trumpsuit, game)
        game.pile.append(card)
        game._record_public_event({
            "type": "card", "player": player.playerNumber, "card": card
        })
        game.cleanCardFromPlayers(card)
        game.incrementTurn()
    before = game.players[pid].real_hand
    current_pile = game.pile.copy()
    legal = game.players[pid].getRealPossibleMoves(current_pile)
    chosen = game.players[pid].makeMove(current_pile, game.trumpsuit, game)
    assert chosen in legal
    assert chosen not in game.players[pid].real_hand
    assert len(game.players[pid].real_hand) == len(before) - 1


@pytest.mark.parametrize("cls", [MonteCarloPlayer, SnakeMonteCarloPlayer])
def test_compatibility_card_monte_carlo_interface(cls, capsys):
    game = game_with_mc_player(cls, 1)
    # Player 1 starts because the initial dealer is 0.
    assert game.turn == 1
    player = game.players[1]
    legal = player.getRealPossibleMoves(game.pile)
    scores, errors = player.cardPlayMonteCarlo(legal[0], 24, game)
    assert errors == 0
    assert len(scores) == 3
    assert all(0 <= p <= 13 for p in scores)


def test_monte_carlo_engine_does_not_mutate_caller_game(capsys):
    game = game_with_mc_player(MonteCarloPlayer, 1)
    before = game.observe(1).to_json()
    before_cards = [p.real_hand for p in game.players]
    evaluate_actions(game.observe(1), n_rollouts=40, seed=104)
    assert [p.real_hand for p in game.players] == before_cards
    assert game.observe(1).to_json() == before


def test_two_bots_can_share_same_game_without_shared_rollout_state(capsys):
    game = game_with_mc_player(MonteCarloPlayer, 1, ncards=2)
    cards = game.players[2].real_hand
    game.players[2] = SnakeMonteCarloPlayer(
        2, cards, additionalParameters={"monteCarloNumber": "12"}
    )
    game.playFullTrick()
    assert game.trickNumber == 1
    game.playPartialRound()
    assert game.roundnumber == 1
    assert sum(game.tricksWon) == 2
