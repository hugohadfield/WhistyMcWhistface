"""Matched-deal complete-game tournament of bid_aware_3 vs bid_aware_2/random.

Every agent has identical RandomPlayer bidding and trump selection. Only
card play differs. Rotate the challenger through every seat on the SAME
deal and bootstrap at the independent-deal level, not the game level.

Examples:
 python -m benchmarks.bid_aware_3_tournament --baseline aware2 --players 3 --games 600
 python -m benchmarks.bid_aware_3_tournament --baseline random --players 2 --games 400
"""
import argparse
import contextlib
import csv
from concurrent.futures import ProcessPoolExecutor
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
from BidAware3Player import BidAware3Player
from RandomPlayer import RandomPlayer

SCHEDULE = (7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7)
BASELINES = {"aware2": BidAware2Player, "random": RandomPlayer}


@dataclass(frozen=True)
class Match:
    game: int
    deal_group: int
    deal_seed: int
    players: int
    challenger_seat: int
    challenger_score: int
    baseline_scores: tuple[int, ...]
    challenger_exact_bids: int
    baseline_exact_bids: int
    challenger_overtricks: int
    baseline_overtricks: int
    challenger_undertricks: int
    baseline_undertricks: int
    three_card_choices: int
    seconds: float

    @property
    def first_share(self):
        scores = (self.challenger_score, *self.baseline_scores)
        best = max(scores)
        return ((1 / sum(x == best for x in scores))
                if self.challenger_score == best else 0)

    @property
    def margin(self):
        return self.challenger_score - max(self.baseline_scores)


def play_match(index, players=3, baseline="aware2", seed=20261120):
    if not 2 <= players <= 7 or index < 0 or baseline not in BASELINES:
        raise ValueError("Invalid players, index or baseline")
    deal_group, seat = divmod(index, players)
    deal_seed = seed + deal_group
    game = Game.__new__(Game)
    game.cardnumbers = list(SCHEDULE)
    game.playerConfigFileName = ""
    game.numberofplayers = players
    game.players = [BidAware3Player(i) if i == seat else BASELINES[baseline](i)
                    for i in range(players)]
    game.resetGame()

    original_deal = game.randomDeal
    def seeded_deal(cards):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return original_deal(cards)
    game.randomDeal = seeded_deal

    bid_hits = [0] * players
    over = [0] * players
    under = [0] * players
    score = game.processScore
    def record_score():
        score()
        for i in range(players):
            diff = game.tricksWon[i] - game.bids[i]
            bid_hits[i] += diff == 0
            over[i] += diff > 0
            under[i] += diff < 0
    game.processScore = record_score

    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    if game.roundnumber != len(SCHEDULE):
        raise AssertionError("Incomplete full game")
    others = [i for i in range(players) if i != seat]
    return Match(
        game=index+1, deal_group=deal_group, deal_seed=deal_seed, players=players,
        challenger_seat=seat, challenger_score=game.scores[seat],
        baseline_scores=tuple(game.scores[i] for i in others),
        challenger_exact_bids=bid_hits[seat],
        baseline_exact_bids=sum(bid_hits[i] for i in others),
        challenger_overtricks=over[seat],
        baseline_overtricks=sum(over[i] for i in others),
        challenger_undertricks=under[seat],
        baseline_undertricks=sum(under[i] for i in others),
        three_card_choices=getattr(game.players[seat], "three_card_choices", 0),
        seconds=time.perf_counter()-started)


def summarize(matches, *, seed=20261120, bootstrap=2000):
    if not matches or bootstrap < 100:
        raise ValueError("Need matches and at least 100 bootstrap replicates")
    players = matches[0].players
    if len(matches) % players:
        raise ValueError("Must have complete seat-rotation deal groups")
    groups = []
    for i in range(0, len(matches), players):
        group = matches[i:i+players]
        if (any(m.deal_group != group[0].deal_group
                or m.deal_seed != group[0].deal_seed or m.players != players
                for m in group)
                or {m.challenger_seat for m in group} != set(range(players))):
            raise ValueError("Invalid seat-rotation group")
        groups.append(group)
    first_by_deal = [statistics.mean(m.first_share for m in g) for g in groups]
    margin_by_deal = [statistics.mean(m.margin for m in g) for g in groups]
    rng = random.Random(seed ^ 0xB1DA33)
    boot_first, boot_margin = [], []
    for _ in range(bootstrap):
        sample = [rng.randrange(len(groups)) for _ in groups]
        boot_first.append(statistics.mean(first_by_deal[i] for i in sample))
        boot_margin.append(statistics.mean(margin_by_deal[i] for i in sample))
    boot_first.sort()
    boot_margin.sort()
    low = int(.025*bootstrap)
    high = min(bootstrap-1,int(.975*bootstrap))
    first = statistics.mean(m.first_share for m in matches)
    n = len(matches)
    return {
        "games": n, "independent_deals": len(groups), "players": players,
        "challenger_first_place_share": first,
        "baseline_first_place_share_per_player": (1-first)/(players-1),
        "neutral_first_place_share": 1/players,
        "challenger_first_place_95_ci": [boot_first[low],boot_first[high]],
        "challenger_outright_wins": sum(
            m.challenger_score > max(m.baseline_scores) for m in matches),
        "challenger_mean_score": statistics.mean(m.challenger_score for m in matches),
        "baseline_mean_score_per_player": statistics.mean(
            x for m in matches for x in m.baseline_scores),
        "challenger_mean_margin_to_best": statistics.mean(margin_by_deal),
        "challenger_mean_margin_95_ci": [boot_margin[low], boot_margin[high]],
        "challenger_exact_bid_rate": sum(m.challenger_exact_bids for m in matches)/(13*n),
        "baseline_exact_bid_rate": sum(m.baseline_exact_bids for m in matches)/
                                   (13*n*(players-1)),
        "challenger_overtrick_rate": sum(m.challenger_overtricks for m in matches)/(13*n),
        "baseline_overtrick_rate": sum(m.baseline_overtricks for m in matches)/
                                  (13*n*(players-1)),
        "challenger_undertrick_rate": sum(m.challenger_undertricks for m in matches)/(13*n),
        "baseline_undertrick_rate": sum(m.baseline_undertricks for m in matches)/
                                   (13*n*(players-1)),
        "three_card_choices_per_game": statistics.mean(
            m.three_card_choices for m in matches),
    }


def tournament(*, baseline="aware2", players=3, games=600, seed=20261120,
               workers=4, bootstrap=2000, progress_every=60,
               json_path=None, csv_path=None):
    if (baseline not in BASELINES or players < 2 or players > 7 or
            games < players or games % players or workers < 1):
        raise ValueError("Games must form complete seat-rotation groups; check inputs")
    start = time.perf_counter()
    results = []
    if workers == 1:
        pool = None
        iterator = (play_match(i, players, baseline, seed) for i in range(games))
    else:
        pool = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=mp.get_context("spawn"))
        iterator = pool.map(
            play_match, range(games), repeat(players), repeat(baseline),
            repeat(seed), chunksize=1)
    try:
        for result in iterator:
            results.append(result)
            if progress_every and len(results) % progress_every == 0:
                size = len(results) // players * players
                if size:
                    s = summarize(results[:size], seed=seed, bootstrap=200)
                    print(
                        f"PROGRESS {len(results)}/{games} "
                        f"first={s['challenger_first_place_share']:.1%} "
                        f"baseline_first={s['baseline_first_place_share_per_player']:.1%} "
                        f"mean_margin={s['challenger_mean_margin_to_best']:+.2f} "
                        f"exact_bids={s['challenger_exact_bid_rate']:.1%}/"
                        f"{s['baseline_exact_bid_rate']:.1%} "
                        f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    summary = summarize(results, seed=seed, bootstrap=bootstrap)
    summary.update({
        "challenger": "bid_aware_3", "baseline": baseline,
        "bidding_policy": "RandomPlayer.makeAverageBid",
        "trump_policy": "RandomPlayer.pick_trumps",
        "seed":seed, "workers":workers,
        "wall_seconds":time.perf_counter()-start,
    })
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for record in results:
                row = asdict(record)
                row["baseline_scores"] = json.dumps(row["baseline_scores"])
                writer.writerow(row)
    print("TOURNAMENT "+json.dumps(summary,sort_keys=True),flush=True)
    return results, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",choices=tuple(BASELINES),default="aware2")
    p.add_argument("--players",type=int,default=3)
    p.add_argument("--games",type=int,default=600)
    p.add_argument("--seed",type=int,default=20261120)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--progress-every",type=int,default=60)
    p.add_argument("--json",type=Path)
    p.add_argument("--csv",type=Path)
    a=p.parse_args()
    tournament(baseline=a.baseline,players=a.players,games=a.games,
               seed=a.seed,workers=a.workers,bootstrap=a.bootstrap,
               progress_every=a.progress_every,
               json_path=a.json,csv_path=a.csv)


if __name__=="__main__":
    main()
