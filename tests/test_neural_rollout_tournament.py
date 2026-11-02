"""Playable neural-guided MC bot and paired head-to-head regression."""
import contextlib
import io

import pytest

torch = pytest.importorskip("torch")

from alphawhist.core.WhistStructure import Game
from alphawhist.learning.checkpoint import save_checkpoint
from alphawhist.learning.model import DistributionalWhistMLP
from NeuralRolloutMonteCarloPlayer import NeuralRolloutMonteCarloPlayer
from benchmarks.neural_rollout_tournament import play_match, summarize, tournament


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(332)
    target = tmp_path_factory.mktemp("neural-mc") / "small.pt"
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    save_checkpoint(target, model, metadata={"teacher_policy": "bid_aware_2"})
    return target


def test_neural_guided_bot_plays_complete_round_legally(checkpoint):
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [2]
    game.players[0] = NeuralRolloutMonteCarloPlayer(
        0, checkpoint=checkpoint, rollouts=3)
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber == 1
    assert len(game.scores) == 3
    assert sum(game.tricksWon) == 2


def test_neural_guided_player_rejects_bad_configuration(checkpoint):
    with pytest.raises(ValueError, match="checkpoint"):
        NeuralRolloutMonteCarloPlayer(0)
    with pytest.raises(ValueError, match="positive"):
        NeuralRolloutMonteCarloPlayer(0, checkpoint=checkpoint, rollouts=0)
    with pytest.raises(ValueError, match="opponent"):
        NeuralRolloutMonteCarloPlayer(
            0, checkpoint=checkpoint, opponent_policy="neural")


def test_matched_seat_swapped_tournament_completes_and_writes_results(
        checkpoint, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        results, report = tournament(
            checkpoint, games=2, seed=9281971, rollouts=2,
            workers=1, baseline="neural", csv_path=tmp_path / "games.csv",
            json_path=tmp_path / "results.json",
            bootstrap=100, progress_every=0,
        )
    assert len(results) == 2
    assert report["paired_deals"] == 1
    assert results[0].deal_seed == results[1].deal_seed
    assert results[0].challenger_seat == results[1].baseline_seat
    assert results[1].challenger_seat == results[0].baseline_seat
    assert report["mean_challenger_minus_baseline"] == (
        results[0].margin + results[1].margin) / 2
    assert report["challenger_head_to_head_wins"] + (
        report["baseline_head_to_head_wins"] + report["head_to_head_ties"]) == 2
    assert report["simulated_opponent_policy"] == "bid_aware_2"
    assert (tmp_path / "games.csv").exists()
    assert (tmp_path / "results.json").exists()
