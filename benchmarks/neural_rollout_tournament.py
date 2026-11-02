"""Matched-deal tournament: neural-guided Monte Carlo versus a frozen baseline.

The challenger uses neural future play for itself. With --opponent-policy
neural it also uses the same model for each simulated opponent, independently
observing only that opponent's private sampled cards and public information.
Default simulated opponents use bid_aware_2. The real third player is random.

Baseline options:
  neural: direct greedy neural card play from the SAME checkpoint (default)
  mc:     classic Monte Carlo card play with bid_aware_2 continuations

Each independent deal seed produces two full 13-round games with the
challenger and baseline switching seats. Report pair-bootstrap confidence
intervals, not standard errors over individual games.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
from dataclasses import asdict, dataclass
from itertools import repeat
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from NeuralValuePlayer import NeuralValuePlayer
from NeuralRolloutMonteCarloPlayer import NeuralRolloutMonteCarloPlayer
from RandomPlayer import RandomPlayer

SEAT_PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Result:
    game: int
    pair: int
    deal_seed: int
    challenger_seat: int
    baseline_seat: int
    random_seat: int
    challenger_score: int
    baseline_score: int
    random_score: int
    margin: int
    seconds: float


def play_match(index, checkpoint, seed=90061101, rollouts=32,
               baseline="neural", baseline_rollouts=128,
               opponent_policy="bid_aware_2"):
    if index < 0 or rollouts < 1 or baseline_rollouts < 1:
        raise ValueError("Invalid game index or rollout count")
    if baseline not in ("neural", "mc"):
        raise ValueError("Unknown baseline")
    if opponent_policy not in ("random", "highest", "bid_aware_2", "neural"):
        raise ValueError("Unknown simulated opponent policy")
    pair = index // 2
    a, b = SEAT_PAIRS[pair % len(SEAT_PAIRS)]
    challenger, opponent = (a, b) if index % 2 == 0 else (b, a)
    third = next(seat for seat in range(3) if seat not in (a, b))
    deal_seed = seed + pair
    random.seed(deal_seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.players[challenger] = NeuralRolloutMonteCarloPlayer(
        challenger, checkpoint=checkpoint, rollouts=rollouts,
        opponent_policy=opponent_policy,
    )
    if baseline == "neural":
        game.players[opponent] = NeuralValuePlayer(
            opponent, checkpoint=checkpoint, device="cpu",
        )
    else:
        game.players[opponent] = MonteCarloPlayer(
            opponent, additionalParameters={
                "monteCarloNumber": str(baseline_rollouts),
                "monteCarloWorkers": "1",
                "monteCarloPolicy": "bid_aware_2",
            },
        )
    assert type(game.players[third]) is RandomPlayer
    original_deal = game.randomDeal

    def seeded_deal(cards):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return original_deal(cards)

    game.randomDeal = seeded_deal
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    if game.roundnumber != len(game.cardnumbers):
        raise AssertionError("Incomplete game")
    return Result(
        game=index + 1, pair=pair, deal_seed=deal_seed,
        challenger_seat=challenger, baseline_seat=opponent,
        random_seat=third,
        challenger_score=game.scores[challenger],
        baseline_score=game.scores[opponent],
        random_score=game.scores[third],
        margin=game.scores[challenger] - game.scores[opponent],
        seconds=time.perf_counter() - started,
    )


def summarize(results, seed=90061101, bootstrap=5000):
    if not results or len(results) % 2 or bootstrap < 100:
        raise ValueError("Need complete paired games and >= 100 bootstrap samples")
    paired = []
    for first, second in zip(results[::2], results[1::2]):
        if (first.pair != second.pair or first.deal_seed != second.deal_seed
                or first.challenger_seat != second.baseline_seat
                or first.baseline_seat != second.challenger_seat):
            raise ValueError("Pair mismatch: deals or swapped seats differ")
        paired.append((first.margin + second.margin) / 2.0)
    rng = random.Random(seed ^ 0xC11A116)
    bootstraps = sorted(
        statistics.mean(rng.choices(paired, k=len(paired)))
        for _ in range(bootstrap)
    )
    return {
        "games": len(results),
        "paired_deals": len(paired),
        "mean_challenger_score": statistics.mean(r.challenger_score for r in results),
        "mean_baseline_score": statistics.mean(r.baseline_score for r in results),
        "mean_random_score": statistics.mean(r.random_score for r in results),
        "mean_challenger_minus_baseline": statistics.mean(paired),
        "paired_bootstrap_95_ci": [
            bootstraps[int(bootstrap * .025)],
            bootstraps[min(bootstrap - 1, int(bootstrap * .975))],
        ],
        "challenger_head_to_head_wins": sum(r.margin > 0 for r in results),
        "baseline_head_to_head_wins": sum(r.margin < 0 for r in results),
        "head_to_head_ties": sum(r.margin == 0 for r in results),
        "challenger_paired_wins": sum(x > 0 for x in paired),
        "baseline_paired_wins": sum(x < 0 for x in paired),
        "paired_ties": sum(x == 0 for x in paired),
        # Stronger tournament objectives than mean own score alone.
        "challenger_outright_wins": sum(
            r.challenger_score > max(r.baseline_score, r.random_score)
            for r in results),
        "baseline_outright_wins": sum(
            r.baseline_score > max(r.challenger_score, r.random_score)
            for r in results),
        "random_outright_wins": sum(
            r.random_score > max(r.challenger_score, r.baseline_score)
            for r in results),
        "challenger_first_place_share": statistics.mean(
            (1.0 / sum(score == high for score in (
                r.challenger_score, r.baseline_score, r.random_score))
             if r.challenger_score == high else 0.0)
            for r in results
            for high in [max(r.challenger_score, r.baseline_score, r.random_score)]
        ),
        "baseline_first_place_share": statistics.mean(
            (1.0 / sum(score == high for score in (
                r.challenger_score, r.baseline_score, r.random_score))
             if r.baseline_score == high else 0.0)
            for r in results
            for high in [max(r.challenger_score, r.baseline_score, r.random_score)]
        ),
        "mean_challenger_minus_best_opponent": statistics.mean(
            r.challenger_score - max(r.baseline_score, r.random_score)
            for r in results
        ),
    }


def tournament(checkpoint, games=20, seed=90061101, rollouts=32,
               workers=4, baseline="neural", baseline_rollouts=128,
               csv_path=None, json_path=None, bootstrap=5000,
               progress_every=2, opponent_policy="bid_aware_2"):
    if games < 2 or games % 2 or workers < 1:
        raise ValueError("Games must be positive even, with workers >= 1")
    if opponent_policy not in ("random", "highest", "bid_aware_2", "neural"):
        raise ValueError("Unknown simulated opponent policy")
    from alphawhist.learning.checkpoint import load_checkpoint
    load_checkpoint(checkpoint, device="cpu")
    checkpoint = str(Path(checkpoint).resolve())
    started = time.perf_counter()
    results = []
    if workers == 1:
        iterator = (play_match(i, checkpoint, seed, rollouts,
                               baseline, baseline_rollouts, opponent_policy)
                    for i in range(games))
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"))
        iterator = pool.map(
            play_match, range(games), repeat(checkpoint), repeat(seed),
            repeat(rollouts), repeat(baseline), repeat(baseline_rollouts),
            repeat(opponent_policy), chunksize=1,
        )
    try:
        for result in iterator:
            results.append(result)
            if progress_every and len(results) % progress_every == 0:
                margin = statistics.mean(x.margin for x in results)
                print(f"PROGRESS {len(results)}/{games} "
                      f"running_margin={margin:+.2f} "
                      f"elapsed={time.perf_counter() - started:.1f}s",
                      flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    summary = summarize(results, seed=seed, bootstrap=bootstrap)
    summary.update({
        "checkpoint": checkpoint,
        "challenger": "neural_guided_monte_carlo",
        "challenger_rollouts": rollouts,
        "self_policy": "frozen_neural",
        "simulated_opponent_policy": opponent_policy,
        "baseline": baseline,
        "baseline_rollouts": baseline_rollouts if baseline == "mc" else None,
        "seed": seed, "workers": workers,
        "wall_seconds": time.perf_counter() - started,
    })
    if csv_path is not None:
        dest = Path(csv_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in results)
    if json_path is not None:
        dest = Path(json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2) + "\n",
                        encoding="utf-8")
    print("TOURNAMENT " + json.dumps(summary, sort_keys=True), flush=True)
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=90061101)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline", choices=("neural", "mc"), default="neural")
    parser.add_argument("--baseline-rollouts", type=int, default=128)
    parser.add_argument("--opponent-policy",
                        choices=("random", "highest", "bid_aware_2", "neural"),
                        default="bid_aware_2",
                        help="Simulated opponents inside challenger MC rollouts")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=2)
    args = parser.parse_args()
    tournament(
        args.checkpoint, args.games, args.seed, args.rollouts,
        args.workers, args.baseline, args.baseline_rollouts,
        args.csv, args.json, args.bootstrap, args.progress_every,
        args.opponent_policy,
    )


if __name__ == "__main__":
    main()
