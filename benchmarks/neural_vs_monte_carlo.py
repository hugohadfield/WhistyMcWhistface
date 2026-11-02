"""Paired complete-game benchmark: trained neural card play vs Monte Carlo.

Controls bidding/trump selection: both bots inherit identical MC methods.
For each independent dealt match, play two 13-round games swapping neural
and MC seats, keeping the same initial hands at all 13 rounds. Third player
is random. Analyze PAIRS, not individual games, for uncertainty.

Example:
 python -m benchmarks.neural_vs_monte_carlo --checkpoint runs/best.pt \
     --games 100 --rollouts 1024 --workers 4 \
     --csv runs/tournament.csv --json runs/tournament-summary.json
"""
import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import os
from pathlib import Path
import random
import statistics
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from NeuralValuePlayer import NeuralValuePlayer
from RandomPlayer import RandomPlayer

PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Result:
    game: int
    pair: int
    deal_seed: int
    neural_seat: int
    mc_seat: int
    random_seat: int
    neural_score: int
    mc_score: int
    random_score: int
    margin: int
    seconds: float


def play_match(index, checkpoint, seed=20261101, rollouts=1024,
               mc_policy="bid_aware_2", device="cpu"):
    if index < 0 or rollouts < 1:
        raise ValueError("Bad game index or Monte Carlo budget")
    if mc_policy not in ("random", "highest", "bid_aware_2"):
        raise ValueError("Invalid Monte Carlo continuation policy")
    pair = index // 2
    a, b = PAIRS[pair % len(PAIRS)]
    neural, mc = (a, b) if index % 2 == 0 else (b, a)
    third = next(x for x in range(3) if x not in (a, b))
    deal_seed = seed + pair
    random.seed(deal_seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.players[neural] = NeuralValuePlayer(
        neural, checkpoint=checkpoint, device=device,
    )
    game.players[mc] = MonteCarloPlayer(
        mc, additionalParameters={
            "monteCarloNumber": str(rollouts), "monteCarloWorkers": "1",
            "monteCarloPolicy": mc_policy,
        },
    )
    assert type(game.players[third]) is RandomPlayer
    original_deal = game.randomDeal

    def seeded_deal(cards):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return original_deal(cards)

    game.randomDeal = seeded_deal
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as fp:
        with contextlib.redirect_stdout(fp):
            game.playFullGame()
    if game.roundnumber != len(game.cardnumbers):
        raise AssertionError("Incomplete match")
    return Result(
        game=index + 1, pair=pair, deal_seed=deal_seed,
        neural_seat=neural, mc_seat=mc, random_seat=third,
        neural_score=game.scores[neural], mc_score=game.scores[mc],
        random_score=game.scores[third],
        margin=game.scores[neural] - game.scores[mc],
        seconds=time.perf_counter() - started,
    )


def summarize(results, seed=20261101, bootstrap=5000):
    if not results or len(results) % 2 or bootstrap < 100:
        raise ValueError("Need even, complete paired matches")
    pairs = []
    for a, b in zip(results[::2], results[1::2]):
        if (a.pair != b.pair or a.deal_seed != b.deal_seed
                or a.neural_seat != b.mc_seat
                or a.mc_seat != b.neural_seat):
            raise ValueError("Invalid paired matched-deal seat swap")
        pairs.append((a.margin + b.margin) / 2)
    rng = random.Random(seed ^ 0xB00415)
    averages = sorted(
        sum(rng.choices(pairs, k=len(pairs))) / len(pairs)
        for _ in range(bootstrap)
    )
    return {
        "games": len(results),
        "paired_deals": len(pairs),
        "mean_neural_score": statistics.mean(x.neural_score for x in results),
        "mean_mc_score": statistics.mean(x.mc_score for x in results),
        "mean_random_score": statistics.mean(x.random_score for x in results),
        "mean_neural_minus_mc": statistics.mean(pairs),
        "paired_bootstrap_95_ci": [
            averages[int(.025 * bootstrap)],
            averages[min(bootstrap - 1, int(.975 * bootstrap))],
        ],
        "neural_head_to_head_wins": sum(x.margin > 0 for x in results),
        "mc_head_to_head_wins": sum(x.margin < 0 for x in results),
        "head_to_head_ties": sum(x.margin == 0 for x in results),
        "neural_paired_wins": sum(x > 0 for x in pairs),
        "mc_paired_wins": sum(x < 0 for x in pairs),
        "paired_ties": sum(x == 0 for x in pairs),
    }


def tournament(checkpoint, games=100, seed=20261101, rollouts=1024,
               workers=1, mc_policy="bid_aware_2", device="cpu",
               csv_path=None, json_path=None, bootstrap=5000):
    if games < 2 or games % 2 or workers < 1:
        raise ValueError("Games must be an even positive number; workers >= 1")
    from alphawhist.learning.checkpoint import load_checkpoint
    # Fail early if the checkpoint is wrong, instead of spawning many workers.
    load_checkpoint(checkpoint, device="cpu")
    started = time.perf_counter()
    if workers == 1:
        iterator = (play_match(i, str(checkpoint), seed, rollouts,
                               mc_policy, device) for i in range(games))
        matches = list(iterator)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            matches = list(pool.map(
                play_match, range(games),
                repeat(str(checkpoint)), repeat(seed), repeat(rollouts),
                repeat(mc_policy), repeat(device), chunksize=1,
            ))
    report = summarize(matches, seed=seed, bootstrap=bootstrap)
    report.update({
        "checkpoint": str(checkpoint), "mc_rollouts": rollouts,
        "mc_policy": mc_policy, "seed": seed, "workers": workers,
        "wall_seconds": time.perf_counter() - started,
    })
    if csv_path is not None:
        dest = Path(csv_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(asdict(matches[0])))
            w.writeheader()
            w.writerows(asdict(x) for x in matches)
    if json_path is not None:
        dest = Path(json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TOURNAMENT " + json.dumps(report, sort_keys=True), flush=True)
    return matches, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mc-policy", choices=("random", "highest", "bid_aware_2"),
                        default="bid_aware_2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    a = parser.parse_args()
    tournament(a.checkpoint, a.games, a.seed, a.rollouts, a.workers,
               a.mc_policy, a.device, a.csv, a.json, a.bootstrap)


if __name__ == "__main__":
    main()
