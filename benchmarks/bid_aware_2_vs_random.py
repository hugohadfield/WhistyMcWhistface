"""Matched-deal games: cheap bid_aware_2 real player vs RandomPlayer.

Identical RandomPlayer trump/bid logic for every seat; only the two-card
play choice differs. Each independent 13-round deal is repeated once with
the bid-aware bot in each seat, allowing correct deal-cluster bootstrap CIs.

Examples:
 python -m benchmarks.bid_aware_2_vs_random --players 2 --games 400 --workers 4
 python -m benchmarks.bid_aware_2_vs_random --players 3 --games 600 --workers 4
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
from BidAware2Player import BidAware2Player
from RandomPlayer import RandomPlayer


@dataclass(frozen=True)
class GameResult:
    game: int
    deal_group: int
    deal_seed: int
    players: int
    aware_seat: int
    aware_score: int
    random_scores: tuple[int, ...]
    aware_exact_bids: int
    random_exact_bids: int
    aware_overtricks: int
    random_overtricks: int
    aware_undertricks: int
    random_undertricks: int
    seconds: float

    @property
    def first_place_share(self):
        scores = (self.aware_score, *self.random_scores)
        high = max(scores)
        return (1.0 / sum(s == high for s in scores)
                if self.aware_score == high else 0.0)

    @property
    def margin(self):
        return self.aware_score - max(self.random_scores)


def play_match(index, players=3, seed=20261102):
    if players not in (2, 3) or index < 0:
        raise ValueError("players must be 2 or 3 and index nonnegative")
    group, seat = divmod(index, players)
    deal_seed = seed + group
    game = Game.__new__(Game)
    game.cardnumbers = [7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    game.playerConfigFileName = ""
    game.numberofplayers = players
    game.players = [
        BidAware2Player(i) if i == seat else RandomPlayer(i)
        for i in range(players)
    ]
    game.resetGame()

    # Each role placement sees the same starting cards and round schedule.
    original_deal = game.randomDeal

    def seeded_deal(cards):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return original_deal(cards)

    game.randomDeal = seeded_deal
    bid_hits = [0] * players
    overshoots = [0] * players
    undershoots = [0] * players
    original_score = game.processScore

    def score_and_record():
        original_score()
        for i in range(players):
            diff = game.tricksWon[i] - game.bids[i]
            bid_hits[i] += int(diff == 0)
            overshoots[i] += int(diff > 0)
            undershoots[i] += int(diff < 0)

    game.processScore = score_and_record
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    if game.roundnumber != len(game.cardnumbers):
        raise AssertionError("Did not finish every round")
    return GameResult(
        game=index+1, deal_group=group, deal_seed=deal_seed,
        players=players, aware_seat=seat, aware_score=game.scores[seat],
        random_scores=tuple(game.scores[i] for i in range(players) if i != seat),
        aware_exact_bids=bid_hits[seat],
        random_exact_bids=sum(v for i,v in enumerate(bid_hits) if i!=seat),
        aware_overtricks=overshoots[seat],
        random_overtricks=sum(v for i,v in enumerate(overshoots) if i!=seat),
        aware_undertricks=undershoots[seat],
        random_undertricks=sum(v for i,v in enumerate(undershoots) if i!=seat),
        seconds=time.perf_counter()-started,
    )


def summarize(results, *, seed=20261102, bootstrap=2000):
    if not results or bootstrap < 100:
        raise ValueError("Need results and >= 100 bootstrap samples")
    players = results[0].players
    if len(results) % players:
        raise ValueError("Need complete seat-rotation deal groups")
    groups = []
    for i in range(0, len(results), players):
        group = results[i:i+players]
        if (any(r.deal_group != group[0].deal_group or
                r.deal_seed != group[0].deal_seed or r.players != players
                for r in group) or
                {r.aware_seat for r in group} != set(range(players))):
            raise ValueError("Bad seat-rotation matched group")
        groups.append(group)
    firsts = [statistics.mean(r.first_place_share for r in g) for g in groups]
    margins = [statistics.mean(r.margin for r in g) for g in groups]
    rng = random.Random(seed ^ 0xB1DA2E2)
    boot_firsts = []
    boot_margins = []
    for _ in range(bootstrap):
        sampled = [rng.randrange(len(groups)) for _ in groups]
        boot_firsts.append(sum(firsts[i] for i in sampled)/len(groups))
        boot_margins.append(sum(margins[i] for i in sampled)/len(groups))
    boot_firsts.sort()
    boot_margins.sort()
    lo, hi = int(bootstrap*.025), min(bootstrap-1,int(bootstrap*.975))
    first_share = statistics.mean(r.first_place_share for r in results)
    return {
        "games": len(results),
        "independent_deals": len(groups),
        "players": players,
        "aware_mean_score": statistics.mean(r.aware_score for r in results),
        "random_mean_score_per_player": statistics.mean(
            s for r in results for s in r.random_scores),
        "aware_first_place_share": first_share,
        "random_first_place_share_per_player": (1-first_share)/(players-1),
        "random_only_expected_first_place_share": 1/players,
        "aware_outright_wins": sum(
            r.aware_score > max(r.random_scores) for r in results),
        "aware_first_place_share_95_ci": [boot_firsts[lo],boot_firsts[hi]],
        "aware_mean_margin_vs_best_random": statistics.mean(margins),
        "aware_mean_margin_95_ci": [boot_margins[lo],boot_margins[hi]],
        "aware_bid_hit_rate": sum(r.aware_exact_bids for r in results) / (13*len(results)),
        "random_bid_hit_rate": sum(r.random_exact_bids for r in results) / (
            13*(players-1)*len(results)),
        "aware_overtrick_rate": sum(r.aware_overtricks for r in results) / (13*len(results)),
        "random_overtrick_rate": sum(r.random_overtricks for r in results) / (
            13*(players-1)*len(results)),
        "aware_undertrick_rate": sum(r.aware_undertricks for r in results) / (13*len(results)),
        "random_undertrick_rate": sum(r.random_undertricks for r in results) / (
            13*(players-1)*len(results)),
    }


def tournament(*, games=300, players=3, workers=4, seed=20261102,
               bootstrap=2000, progress_every=50, json_path=None, csv_path=None):
    if players not in (2,3) or games < players or games % players or workers < 1:
        raise ValueError("Games must be a positive multiple of 2/3 players; workers positive")
    start = time.perf_counter()
    results = []
    if workers == 1:
        pool = None
        iterator = (play_match(i, players, seed) for i in range(games))
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        iterator = pool.map(play_match, range(games), repeat(players),
                            repeat(seed), chunksize=1)
    try:
        for result in iterator:
            results.append(result)
            if progress_every and len(results) % progress_every == 0:
                n_complete = len(results) // players * players
                if n_complete:
                    report = summarize(results[:n_complete], seed=seed, bootstrap=200)
                    print(
                        f"PROGRESS {len(results)}/{games} "
                        f"aware_first={report['aware_first_place_share']:.1%} "
                        f"random_first_per_seat={report['random_first_place_share_per_player']:.1%} "
                        f"margin_vs_best_random={report['aware_mean_margin_vs_best_random']:+.2f} "
                        f"bid_hits={report['aware_bid_hit_rate']:.1%}/"
                        f"{report['random_bid_hit_rate']:.1%} "
                        f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    report = summarize(results,seed=seed,bootstrap=bootstrap)
    report.update({
        "aware_policy": "bid_aware_2", "baseline_policy": "random",
        "bidding_policy": "RandomPlayer.makeAverageBid",
        "trump_policy": "RandomPlayer.pick_trumps",
        "seed":seed, "workers":workers, "seconds":time.perf_counter()-start,
    })
    if json_path is not None:
        dest = Path(json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if csv_path is not None:
        dest=Path(csv_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w",newline="",encoding="utf-8") as sink:
            writer=csv.DictWriter(sink,fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                row=asdict(result)
                row["random_scores"]=json.dumps(row["random_scores"])
                writer.writerow(row)
    print("TOURNAMENT "+json.dumps(report,sort_keys=True),flush=True)
    return results, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players",type=int,choices=(2,3),default=3)
    parser.add_argument("--games",type=int,default=300)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=20261102)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--progress-every",type=int,default=50)
    parser.add_argument("--csv",type=Path)
    parser.add_argument("--json",type=Path)
    args=parser.parse_args()
    tournament(games=args.games,players=args.players,workers=args.workers,
               seed=args.seed,bootstrap=args.bootstrap,
               progress_every=args.progress_every,
               json_path=args.json,csv_path=args.csv)


if __name__=="__main__":
    main()
