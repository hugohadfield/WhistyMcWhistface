"""Seat-swapped 13-round test: MonteCarloBidPlayer vs original MC bidder.

Both bots use IDENTICAL MC(1024, bid_aware_2) card play; the only change is
choosing a bid via independent cheap full-round simulations. The third player
uses RandomPlayer. Independent deals are paired across swapped seats.

Example:
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.mc_bid_tournament \\
   --games 100 --workers 4 --play-rollouts 1024 --bid-rollouts 1024 \\
   --json runs/mc-bid-tournament.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
from dataclasses import asdict
from itertools import repeat
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from MonteCarloBidPlayer import MonteCarloBidPlayer
from RandomPlayer import RandomPlayer
from benchmarks.neural_rollout_tournament import (
    Result, summarize, progress_report, SEAT_PAIRS,
)


def play_match(index, seed=20261102, play_rollouts=1024, bid_rollouts=1024):
    if index < 0 or min(play_rollouts, bid_rollouts) < 1:
        raise ValueError("Bad game index or rollout budget")
    pair = index // 2
    a, b = SEAT_PAIRS[pair % len(SEAT_PAIRS)]
    challenger, baseline = (a, b) if index % 2 == 0 else (b, a)
    third = next(seat for seat in range(3) if seat not in (a, b))
    deal_seed = seed + pair
    random.seed(deal_seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    kwargs = {"monteCarloNumber": str(play_rollouts),
              "monteCarloWorkers": "1", "monteCarloPolicy": "bid_aware_2"}
    game.players[challenger] = MonteCarloBidPlayer(
        challenger, additionalParameters={**kwargs, "bidRollouts": str(bid_rollouts)})
    game.players[baseline] = MonteCarloPlayer(
        baseline, additionalParameters=kwargs)
    assert type(game.players[third]) is RandomPlayer
    original_deal = game.randomDeal

    def seeded_deal(cards):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return original_deal(cards)

    game.randomDeal = seeded_deal
    start = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    if game.roundnumber != len(game.cardnumbers):
        raise AssertionError("Incomplete Whist match")
    return Result(
        game=index + 1, pair=pair, deal_seed=deal_seed,
        challenger_seat=challenger, baseline_seat=baseline,
        random_seat=third, challenger_score=game.scores[challenger],
        baseline_score=game.scores[baseline], random_score=game.scores[third],
        margin=game.scores[challenger] - game.scores[baseline],
        seconds=time.perf_counter() - start,
    )


def tournament(*, games=20, seed=20261102, play_rollouts=1024,
               bid_rollouts=1024, workers=4, bootstrap=5000,
               progress_every=2, json_path=None, csv_path=None):
    if games < 2 or games % 2 or workers < 1:
        raise ValueError("Need positive even game count and positive workers")
    start = time.perf_counter()
    results = []
    if workers == 1:
        iterator = (play_match(i, seed, play_rollouts, bid_rollouts)
                    for i in range(games))
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"))
        iterator = pool.map(
            play_match, range(games), repeat(seed),
            repeat(play_rollouts), repeat(bid_rollouts), chunksize=1)
    try:
        for result in iterator:
            results.append(result)
            if progress_every and len(results) % progress_every == 0:
                print(progress_report(
                    results, games, time.perf_counter()-start, seed=seed,
                    bootstrap=min(bootstrap, 500)), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    result = summarize(results, seed=seed, bootstrap=bootstrap)
    result.update({
        "challenger": "mc_bid", "baseline": "mc_original_bid",
        "card_play_policy": "mc_bid_aware_2", "card_play_rollouts": play_rollouts,
        "bid_rollouts": bid_rollouts,
        "seed": seed, "workers": workers,
        "wall_seconds": time.perf_counter()-start,
    })
    if csv_path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf8") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print("TOURNAMENT "+json.dumps(result, sort_keys=True), flush=True)
    return results, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--play-rollouts", type=int, default=1024)
    parser.add_argument("--bid-rollouts", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    tournament(games=args.games, seed=args.seed,
               play_rollouts=args.play_rollouts,
               bid_rollouts=args.bid_rollouts, workers=args.workers,
               bootstrap=args.bootstrap, progress_every=args.progress_every,
               json_path=args.json, csv_path=args.csv)


if __name__ == "__main__":
    main()
