"""End-to-end 13-round tests against random players.

Unlike the short-rollout tests, these go through dealing, trump selection,
bidding, every card decision, scoring and all round transitions.
"""
import contextlib
import io
import random

import pytest

from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer
from RandomPlayer import RandomPlayer


def run_match(bot_class, seat, rollouts=8, seed=42):
    random.seed(seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.players[seat] = bot_class(
        seat, additionalParameters={
            "monteCarloNumber": str(rollouts),
            "monteCarloWorkers": "1",
        },
    )
    observed = []
    original = game.processScore

    def validate_round():
        cards = game.cardnumbers[game.roundnumber]
        assert game.trickNumber == cards
        assert sum(game.tricksWon) == cards
        assert all(len(player.real_hand) == 0 for player in game.players)
        state = game.observe(0)
        assert state.phase == "scoring"
        assert state.trick_number == cards
        assert state.tricks_won == tuple(game.tricksWon)
        assert state.own_hand == ()
        assert all(opponent.remaining_count == 0 for opponent in state.opponents)
        assert len(state.played_cards) == cards * game.numberofplayers
        assert len(set(state.played_cards)) == len(state.played_cards)
        assert all(0 <= bid <= cards for bid in game.bids)
        assert InformationState.from_json(state.to_json()) == state
        observed.append(state)
        original()

    game.processScore = validate_round
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber == len(game.cardnumbers) == 13
    assert len(observed) == 13
    assert game.observe(0).phase == "between_rounds"
    assert len(game.observe(0).public_history) > 200
    assert all(isinstance(x, int) and x >= 0 for x in game.scores)
    return game


@pytest.mark.parametrize("seat", [0, 1, 2])
@pytest.mark.parametrize("bot_class", [MonteCarloPlayer, SnakeMonteCarloPlayer])
def test_full_thirteen_rounds_ai_against_two_random_players(bot_class, seat):
    run_match(bot_class, seat, rollouts=8, seed=100 + seat)


def test_full_thirteen_rounds_random_against_random():
    random.seed(101)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber == 13
    assert game.observe(0).phase == "between_rounds"


def test_full_game_has_49_tricks_and_147_card_plays():
    game = run_match(MonteCarloPlayer, 0, rollouts=4, seed=246)
    history = game.observe(0).public_history
    assert sum(e["type"] == "trick_end" for e in history) == 49
    assert sum(e["type"] == "card" for e in history) == 147
    assert sum(e["type"] == "bid" for e in history) == 39
    assert sum(e["type"] == "round_start" for e in history) == 13
    assert sum(e["type"] == "round_end" for e in history) == 13


def test_full_game_can_be_replayed_from_information_boundaries():
    game = run_match(MonteCarloPlayer, 2, rollouts=4, seed=36)
    info = game.observe(0)
    loaded = InformationState.from_json(info.to_json())
    assert loaded == info
    assert [e["type"] for e in loaded.public_history].count("card") == 147
