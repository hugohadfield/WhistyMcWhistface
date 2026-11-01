"""Tests for exact-PMF trump-choice objective and paired full-game tournament."""
import importlib

import pytest

from alphawhist.core import WhistLib as lib
from benchmarks.trump_tournament import (
    HeuristicTrumpMonteCarloPlayer, play_match, tournament,
)
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer


def fake_card_win_probabilities(hand, nplayers, ncards, trump, samples=512):
    # Hearts: zero tricks certain -> bid 0 has value 10, banned bid 0 has 0.
    # Diamonds: exactly two independent coin-flip opportunities -> value 6.
    # Clubs/spades: two weak 10% cards -> value at most 8.3 with zero allowed.
    p = {"h": 0.0, "d": 0.5, "c": 0.1, "s": 0.1}[trump]
    return [p] * len(hand), [p] * len(hand)


def test_trump_objective_considers_expected_score_and_zero_bid_legality(monkeypatch):
    monkeypatch.setattr(lib, "estimate_card_win_probabilities",
                        fake_card_win_probabilities)
    hand = ("14h", "2d")
    assert lib.best_trump_by_expected_score(hand, 3) == "h"
    assert lib.best_trump_by_expected_score(hand, 3, can_bid_zero=False) == "d"


def test_trump_picker_uses_expected_points_not_only_contract_confidence(monkeypatch):
    # All-heart trumps lead to a guaranteed 0 tricks -> score 10 with bid zero.
    # All-diamond trumps lead to a 95% chance per card of winning both.
    # Diamond contract confidence is lower, but expected score is higher:
    # 2*0.95 + 10*0.95**2 = 10.925 > 10.
    def probs(hand, nplayers, ncards, trump, samples=512):
        p = {"h": 0.0, "d": 0.95, "c": 0.4, "s": 0.4}[trump]
        return [p] * len(hand), [p] * len(hand)

    monkeypatch.setattr(lib, "estimate_card_win_probabilities", probs)
    assert lib.best_trump_by_expected_score(("14h", "3d"), 3) == "d"


@pytest.mark.parametrize(
    "cls,module_name",
    [(MonteCarloPlayer, "MonteCarloPlayer"),
     (SnakeMonteCarloPlayer, "SnakeMonteCarloPlayer")]
)
def test_both_monte_carlo_agents_activate_new_picker(cls, module_name, monkeypatch):
    module = importlib.import_module(module_name)
    captured = {}

    def fake_picker(hand, nplayers, can_bid_zero=True, samples=512):
        captured.update(hand=tuple(hand), players=nplayers,
                        zero_allowed=can_bid_zero)
        return "s"

    monkeypatch.setattr(module, "best_trump_by_expected_score", fake_picker)
    player = cls(0, ["14h", "7s"])
    player.resetZeroCounter(2)
    assert player.pick_trumps(3) == "s"
    assert captured == {
        "hand": ("14h", "7s"), "players": 3, "zero_allowed": False
    }


def test_trump_picker_rejects_bad_hands():
    with pytest.raises(ValueError, match="nonempty"):
        lib.best_trump_by_expected_score([], 3)
    with pytest.raises(ValueError, match="two players"):
        lib.best_trump_by_expected_score(["14h"], 1)


def test_real_trump_picker_returns_suit_for_small_sample():
    trump = lib.best_trump_by_expected_score(
        ["14h", "2d", "6s"], nplayers=3, samples=12
    )
    assert trump in "hcds"


def test_heuristic_comparison_player_retains_original_picker(monkeypatch):
    import benchmarks.trump_tournament as mod
    received = []
    monkeypatch.setattr(
        mod.TrumpPicker, "picktrump", lambda hand: received.append(hand) or "c"
    )
    player = HeuristicTrumpMonteCarloPlayer(0, ["14h", "2s"])
    assert player.pick_trumps(3) == "c"
    assert received == [["14h", "2s"]]


def test_paired_tournament_completes_and_rotates_seats(tmp_path, capsys):
    path = tmp_path / "paired.csv"
    results = tournament(games=2, rollouts=4, seed=7441, csv_path=path)
    assert len(results) == 2
    assert results[0].deal_seed == results[1].deal_seed
    assert (results[0].new_seat, results[0].old_seat) == (0, 1)
    assert (results[1].new_seat, results[1].old_seat) == (1, 0)
    assert results[0].random_seat == results[1].random_seat == 2
    assert all(r.result in ("NEW", "OLD", "TIE") for r in results)
    assert all(r.winner in ("NEW", "OLD", "RANDOM", "TIE_FOR_FIRST")
               for r in results)
    assert all(r.margin == r.new_score - r.old_score for r in results)
    assert "SUMMARY games=2 paired_deals=1" in capsys.readouterr().out
    assert len(path.read_text().splitlines()) == 3


def test_invalid_tournament_parameters():
    with pytest.raises(ValueError, match="even"):
        tournament(games=3, rollouts=4)
    with pytest.raises(ValueError, match="positive"):
        play_match(0, rollouts=0)
    with pytest.raises(ValueError, match="game_index"):
        play_match(-1, rollouts=4)
