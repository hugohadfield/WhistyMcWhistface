"""Whole-game wall-time benchmark: a Monte Carlo bot versus two random bots.

Run from the repository root:
    python -m benchmarks.full_game --rollouts 16 128 1024
    python -m benchmarks.full_game --rollouts 1024 --repeats 3 --seat 0

Reported times include all 13 rounds, 55 tricks, 165 card plays, dealing, bids and scoring.
The default uses one worker and suppresses gameplay printing; --workers overrides it.
"""
import argparse
import contextlib
import os
import random
import statistics
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer


def timed_game(bot=None, rollouts=16, seed=42, seat=0, workers=1):
    random.seed(seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    if bot is not None:
        game.players[seat] = bot(
            seat, additionalParameters={
                "monteCarloNumber": str(rollouts),
                "monteCarloWorkers": str(workers),
            }
        )
    start = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as null_output:
        with contextlib.redirect_stdout(null_output):
            game.playFullGame()
    duration = time.perf_counter() - start
    assert game.roundnumber == 13
    assert game.observe(0).phase == "between_rounds"
    return duration, tuple(game.scores)


def benchmark(rollouts=(16, 128, 1024), repeats=1, seat=0, workers=1,
              bots=("MonteCarloPlayer",)):
    options = {
        "MonteCarloPlayer": MonteCarloPlayer,
        "SnakeMonteCarloPlayer": SnakeMonteCarloPlayer,
    }
    for name, bot, count in [
        ("RandomPlayer", None, 0),
        *((name, options[name], n) for name in bots for n in rollouts),
    ]:
        samples = []
        last_scores = None
        for i in range(repeats):
            elapsed, scores = timed_game(
                bot, count, seed=20261031 + i, seat=seat, workers=workers
            )
            samples.append(elapsed)
            last_scores = scores
        print(
            f"bot={name} rollouts_per_decision={count} "
            f"workers={workers} seat={seat} games={repeats} "
            f"median_sec={statistics.median(samples):.3f} "
            f"min_sec={min(samples):.3f} max_sec={max(samples):.3f} "
            f"last_scores={last_scores}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, nargs="+",
                        default=[16, 128, 1024])
    parser.add_argument("--seat", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--bots", nargs="+", choices=[
        "MonteCarloPlayer", "SnakeMonteCarloPlayer"
    ], default=["MonteCarloPlayer"])
    args = parser.parse_args()
    benchmark(args.rollouts, args.repeats, args.seat, args.workers, args.bots)
