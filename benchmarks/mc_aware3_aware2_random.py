"""Three-way Whist tournament: MC(bid_aware_3), MC(bid_aware_2), RandomPlayer.

This compares TWO IDENTICAL MonteCarloPlayer agents (same bid/trump logic,
same card-play rollout count, same evaluator and score objective) differing
ONLY in their simulated future-card policy. The third seat is RandomPlayer.

Every independent deal is played six times with all 3! role-to-seat
permutations, so each agent occupies each seat twice on identical initial
deals. Bootstrap confidence intervals cluster by independent deal, not game.
Each full match contains the canonical 13 rounds.

Examples:
 python -m benchmarks.mc_aware3_aware2_random --games 360 \\
   --rollouts 256 --workers 4 --seed 20261148 \\
   --json runs/mc-256.json --csv runs/mc-256.csv

 python -m benchmarks.mc_aware3_aware2_random --games 120 \\
   --rollouts 1024 --workers 4 --seed 20261149 \\
   --json runs/mc-1024.json --csv runs/mc-1024.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
from dataclasses import asdict, dataclass
from itertools import permutations, repeat
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from RandomPlayer import RandomPlayer

# Each permutation lists the seats for (MC3, MC2, random).
ROLE_TO_SEAT = tuple(permutations(range(3)))
ROLE_NAMES = ("mc_bid_aware_3", "mc_bid_aware_2", "random")


@dataclass(frozen=True)
class Match:
    game: int
    deal_group: int
    deal_seed: int
    seats: tuple[int, int, int]
    scores: tuple[int, int, int]
    exact_bids: tuple[int, int, int]
    overtricks: tuple[int, int, int]
    undertricks: tuple[int, int, int]
    seconds: float

    @property
    def first_place_shares(self):
        high = max(self.scores)
        n_tied = self.scores.count(high)
        return tuple(1.0/n_tied if score == high else 0.0
                     for score in self.scores)


def play_match(index, *, seed=20261148, rollouts=256):
    if index < 0 or rollouts < 1:
        raise ValueError("Game index nonnegative and MC rollouts positive")
    group, rotation = divmod(index, len(ROLE_TO_SEAT))
    seats = ROLE_TO_SEAT[rotation]
    deal_seed = seed+group
    random.seed(deal_seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    params = {"monteCarloNumber":str(rollouts),
              "monteCarloWorkers":"1"}
    game.players[seats[0]] = MonteCarloPlayer(
        seats[0], additionalParameters={
            **params, "monteCarloPolicy":"bid_aware_3"})
    game.players[seats[1]] = MonteCarloPlayer(
        seats[1], additionalParameters={
            **params, "monteCarloPolicy":"bid_aware_2"})
    assert type(game.players[seats[2]]) is RandomPlayer

    # Reset per-round DEAL RNG to the same independent hands for every
    # permutation, notwithstanding different stochastic card-play histories.
    original_deal = game.randomDeal
    def paired_deal(cards):
        random.seed(deal_seed*1009 + game.roundnumber)
        return original_deal(cards)
    game.randomDeal = paired_deal

    exact = [0]*3
    over = [0]*3
    under = [0]*3
    old_score = game.processScore
    def record_score():
        old_score()
        for seat in range(3):
            delta = game.tricksWon[seat]-game.bids[seat]
            exact[seat] += int(delta == 0)
            over[seat] += int(delta > 0)
            under[seat] += int(delta < 0)
    game.processScore = record_score
    start = time.perf_counter()
    with open(os.devnull,"w",encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    if game.roundnumber != len(game.cardnumbers):
        raise AssertionError("Incomplete 13-round match")
    if any(exact[i]+over[i]+under[i]!=len(game.cardnumbers)
           for i in range(3)):
        raise AssertionError("Incomplete bid outcomes")
    return Match(
        game=index+1, deal_group=group, deal_seed=deal_seed,
        seats=seats,
        scores=tuple(game.scores[seat] for seat in seats),
        exact_bids=tuple(exact[seat] for seat in seats),
        overtricks=tuple(over[seat] for seat in seats),
        undertricks=tuple(under[seat] for seat in seats),
        seconds=time.perf_counter()-start,
    )


def _confidence_interval(items, rng, *, bootstrap):
    boots=[]
    for _ in range(bootstrap):
        boots.append(statistics.mean(rng.choices(items,k=len(items))))
    boots.sort()
    return [boots[int(.025*bootstrap)],
            boots[min(bootstrap-1,int(.975*bootstrap))]]


def summarize(matches, *, seed=20261148, bootstrap=3000):
    if not matches or len(matches)%6 or bootstrap<100:
        raise ValueError("Require complete six-seat deal groups and >=100 bootstrap samples")
    ordered = sorted(matches, key=lambda match:(match.deal_group,
                                                ROLE_TO_SEAT.index(match.seats)))
    groups=[]
    for first in range(0,len(ordered),6):
        group=ordered[first:first+6]
        if (len({m.deal_group for m in group})!=1 or
            len({m.deal_seed for m in group})!=1 or
            {m.seats for m in group}!=set(ROLE_TO_SEAT)):
            raise ValueError("Incomplete or inconsistent seat-permutation group")
        groups.append(group)
    n=len(matches)
    def avg(f):
        return statistics.mean(f(m) for m in matches)
    shares=[avg(lambda m, i=i:m.first_place_shares[i])
            for i in range(3)]
    scores=[avg(lambda m,i=i:m.scores[i])
            for i in range(3)]
    exact=[sum(m.exact_bids[i] for m in matches)/(13*n)
           for i in range(3)]
    over=[sum(m.overtricks[i] for m in matches)/(13*n)
          for i in range(3)]
    under=[sum(m.undertricks[i] for m in matches)/(13*n)
           for i in range(3)]
    head_to_head = [
        sum(m.scores[0]>m.scores[1] for m in matches),
        sum(m.scores[0]<m.scores[1] for m in matches),
        sum(m.scores[0]==m.scores[1] for m in matches),
    ]
    rng=random.Random(seed ^ 0xB1DAA33)
    deal_mean_diffs = [
        statistics.mean(m.scores[0]-m.scores[1] for m in group)
        for group in groups]
    deal_first_diffs = [
        statistics.mean(m.first_place_shares[0]-m.first_place_shares[1]
                        for m in group)
        for group in groups]
    deal_share = [
        [statistics.mean(m.first_place_shares[i] for m in group)
         for group in groups]
        for i in range(3)
    ]
    return {
        "games":n, "independent_deals":len(groups),
        "rotations_per_deal":6, "players":3,
        "roles":ROLE_NAMES,
        "mean_scores_by_role":dict(zip(ROLE_NAMES,scores)),
        "first_place_shares_by_role":dict(zip(ROLE_NAMES,shares)),
        "first_place_share_95_ci_by_role":{
            name:_confidence_interval(deal_share[i],rng,bootstrap=bootstrap)
            for i,name in enumerate(ROLE_NAMES)},
        "exact_bid_rates_by_role":dict(zip(ROLE_NAMES,exact)),
        "overtrick_rates_by_role":dict(zip(ROLE_NAMES,over)),
        "undertrick_rates_by_role":dict(zip(ROLE_NAMES,under)),
        "mc3_minus_mc2_mean_score":statistics.mean(deal_mean_diffs),
        "mc3_minus_mc2_score_95_ci":
            _confidence_interval(deal_mean_diffs,rng,bootstrap=bootstrap),
        "mc3_minus_mc2_first_share":
            statistics.mean(deal_first_diffs),
        "mc3_minus_mc2_first_share_95_ci":
            _confidence_interval(deal_first_diffs,rng,bootstrap=bootstrap),
        "mc3_vs_mc2_head_to_head":dict(zip(
            ("mc3_wins","mc2_wins","ties"),head_to_head)),
        "outright_wins_by_role":dict(zip(
            ROLE_NAMES,
            [sum(m.scores[i]>max(m.scores[j] for j in range(3) if j!=i)
                 for m in matches) for i in range(3)])),
        "mean_game_seconds":avg(lambda m:m.seconds),
    }


def tournament(*, games=360, rollouts=256, seed=20261148,
               workers=4, bootstrap=3000, progress_every=30,
               json_path=None, csv_path=None):
    if games<6 or games%6 or rollouts<1 or workers<1 or bootstrap<100:
        raise ValueError("Games multiple of six >=6; budgets and workers positive")
    started=time.perf_counter()
    matches=[]
    if workers==1:
        pool=None
        iterator=(play_match(i,seed=seed,rollouts=rollouts)
                  for i in range(games))
    else:
        pool=ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn"))
        # play_match has keyword-only args, so wrapper for process workers.
        iterator=pool.map(_worker_play_match, range(games),
                          repeat(seed),repeat(rollouts),chunksize=1)
    try:
        for match in iterator:
            matches.append(match)
            if progress_every and len(matches)%progress_every==0:
                n=len(matches)//6*6
                if n:
                    s=summarize(matches[:n],seed=seed,bootstrap=200)
                    print(
                        f"PROGRESS {len(matches)}/{games} "
                        f"first={s['first_place_shares_by_role']} "
                        f"mean_score_diff={s['mc3_minus_mc2_mean_score']:+.3f} "
                        f"elapsed={time.perf_counter()-started:.1f}s",
                        flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    summary=summarize(matches,seed=seed,bootstrap=bootstrap)
    summary.update({"rollouts_per_mc_card_decision":rollouts,
                    "mc_search_objective":"expected own round points",
                    "seed":seed, "workers":workers,
                    "wall_seconds":time.perf_counter()-started})
    if json_path is not None:
        path=Path(json_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(summary,indent=2)+"\n",
                        encoding="utf-8")
    if csv_path is not None:
        path=Path(csv_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="",encoding="utf-8") as sink:
            writer=csv.DictWriter(sink,fieldnames=list(asdict(matches[0])))
            writer.writeheader()
            for match in matches:
                row=asdict(match)
                for k in ("seats","scores","exact_bids","overtricks","undertricks"):
                    row[k]=json.dumps(row[k])
                writer.writerow(row)
    print("TOURNAMENT "+json.dumps(summary,sort_keys=True),flush=True)
    return matches,summary


def _worker_play_match(index, seed, rollouts):
    return play_match(index,seed=seed,rollouts=rollouts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games",type=int,default=360)
    parser.add_argument("--rollouts",type=int,default=256)
    parser.add_argument("--seed",type=int,default=20261148)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--bootstrap",type=int,default=3000)
    parser.add_argument("--progress-every",type=int,default=30)
    parser.add_argument("--json",type=Path)
    parser.add_argument("--csv",type=Path)
    a=parser.parse_args()
    tournament(games=a.games,rollouts=a.rollouts,seed=a.seed,
               workers=a.workers,bootstrap=a.bootstrap,
               progress_every=a.progress_every,
               json_path=a.json,csv_path=a.csv)


if __name__=="__main__":
    main()
