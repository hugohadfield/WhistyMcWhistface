"""Head-to-head: bid-aware 2-card continuation versus random continuation.

The ONLY intentional difference between the two MonteCarloPlayer instances is
the policy used inside future-card rollouts. Both get identical rollout budget,
expected-score trump picker, bid predictor and immediate card selection.

250 games = 125 same-deal seat-swap pairs, with the third seat RandomPlayer.
Each game is a complete 13-round, 3-player match. This is a 3-player
competitive evaluation, NOT a claim about strength at other player counts.

Run:
 python -m benchmarks.endgame_policy_tournament --games 250 --rollouts 1024 \
     --workers 4 --seed 20261301 --csv tournament.csv
"""
import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
from pathlib import Path
import random
import statistics
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from RandomPlayer import RandomPlayer

PAIR_SEATS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Match:
    game: int
    pair: int
    deal_seed: int
    aware_seat: int
    random_policy_seat: int
    random_opponent_seat: int
    aware_score: int
    random_policy_score: int
    random_opponent_score: int
    margin: int
    h2h: str
    outright: str
    seconds: float


def play_match(game_index, seed=20261301, rollouts=1024, training_dir=None):
    if game_index < 0 or rollouts < 1:
        raise ValueError("Game index must be nonnegative; rollouts positive")
    pair = game_index // 2
    left, right = PAIR_SEATS[pair % len(PAIR_SEATS)]
    aware_seat, random_policy_seat = (
        (left, right) if game_index % 2 == 0 else (right, left)
    )
    third = ({0, 1, 2} - {left, right}).pop()
    deal_seed = seed + pair
    random.seed(deal_seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    common = {"monteCarloNumber": str(rollouts), "monteCarloWorkers": "1"}
    game.players[aware_seat] = MonteCarloPlayer(
        aware_seat, additionalParameters={
            **common, "monteCarloPolicy": "bid_aware_2"
        }
    )
    game.players[random_policy_seat] = MonteCarloPlayer(
        random_policy_seat, additionalParameters={
            **common, "monteCarloPolicy": "random"
        }
    )
    assert isinstance(game.players[third], RandomPlayer)
    assert type(game.players[aware_seat]) is type(game.players[random_policy_seat])
    assert game.players[aware_seat].monteCarloNumber == (
        game.players[random_policy_seat].monteCarloNumber
    )
    # Re-seed every round's shuffled deal, independent of previous policies.
    ordinary_deal = game.randomDeal

    def fixed_round_deal(cards_in_round):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return ordinary_deal(cards_in_round)

    game.randomDeal = fixed_round_deal
    started = time.perf_counter()
    # Each match owns its own trace file: safe across parallel game workers.
    if training_dir is not None:
        from alphawhist.learning.recording import TrainingRecorder
        training_dir = Path(training_dir)
        training_dir.mkdir(parents=True, exist_ok=True)
        recorder_context = TrainingRecorder(
            training_dir / f"match-{game_index:07d}.jsonl",
            game_id=f"tournament-{seed}-pair-{pair}-seat-swap-{game_index % 2}",
            metadata={
                "seed": seed, "deal_seed": deal_seed, "game_index": game_index,
                "rollouts": rollouts, "agents": {
                    "aware": aware_seat, "random_policy": random_policy_seat,
                    "random_opponent": third,
                },
            },
        )
    else:
        recorder_context = contextlib.nullcontext(None)
    with recorder_context as recorder:
        if recorder is not None:
            game.training_recorder = recorder
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                game.playFullGame()
    elapsed = time.perf_counter() - started
    assert game.roundnumber == 13
    history = game.observe(0).public_history
    assert sum(e["type"] == "round_end" for e in history) == 13
    assert sum(e["type"] == "trick_end" for e in history) == 55
    assert sum(e["type"] == "card" for e in history) == 165

    aware = game.scores[aware_seat]
    ordinary = game.scores[random_policy_seat]
    outsider = game.scores[third]
    margin = aware - ordinary
    top = max(game.scores)
    winners = [i for i, score in enumerate(game.scores) if score == top]
    outright = ("AWARE" if winners == [aware_seat] else
                "RANDOM_POLICY" if winners == [random_policy_seat] else
                "RANDOM_OPPONENT" if winners == [third] else "TIE_FOR_FIRST")
    return Match(
        game=game_index + 1, pair=pair + 1, deal_seed=deal_seed,
        aware_seat=aware_seat, random_policy_seat=random_policy_seat,
        random_opponent_seat=third,
        aware_score=aware, random_policy_score=ordinary,
        random_opponent_score=outsider, margin=margin,
        h2h=("AWARE" if margin > 0 else "RANDOM_POLICY" if margin < 0 else "TIE"),
        outright=outright, seconds=elapsed,
    )


def summary(results, seed=20261301, bootstrap=20000):
    """Bootstrap whole seat-swapped deal PAIRS, never individual games."""
    if not results or len(results) % 2:
        raise ValueError("Need complete same-deal seat-swap pairs")
    if any(results[i].deal_seed != results[i+1].deal_seed
           or results[i].aware_seat != results[i+1].random_policy_seat
           or results[i].random_policy_seat != results[i+1].aware_seat
           for i in range(0, len(results), 2)):
        raise ValueError("Results are not ordered, matched seat-swap pairs")
    paired = [
        (results[i].margin + results[i+1].margin) / 2
        for i in range(0, len(results), 2)
    ]
    rng = random.Random(seed ^ 0x77BEEF)
    boot = sorted(
        sum(rng.choices(paired, k=len(paired))) / len(paired)
        for _ in range(bootstrap)
    )
    lo = boot[int(bootstrap * 0.025)]
    hi = boot[int(bootstrap * 0.975)]
    return {
        "games": len(results), "paired_deals": len(paired),
        "aware_wins": sum(m.h2h == "AWARE" for m in results),
        "random_policy_wins": sum(m.h2h == "RANDOM_POLICY" for m in results),
        "ties": sum(m.h2h == "TIE" for m in results),
        "outright_aware": sum(m.outright == "AWARE" for m in results),
        "outright_random_policy": sum(m.outright == "RANDOM_POLICY" for m in results),
        "outright_random_opponent": sum(m.outright == "RANDOM_OPPONENT" for m in results),
        "outright_ties": sum(m.outright == "TIE_FOR_FIRST" for m in results),
        "average_aware": statistics.mean(m.aware_score for m in results),
        "average_random_policy": statistics.mean(m.random_policy_score for m in results),
        "average_random_opponent": statistics.mean(m.random_opponent_score for m in results),
        "average_margin": statistics.mean(paired),
        "median_margin": statistics.median(m.margin for m in results),
        "pairs_aware": sum(x > 0 for x in paired),
        "pairs_random_policy": sum(x < 0 for x in paired),
        "pairs_ties": sum(x == 0 for x in paired),
        "bootstrap_95": (lo, hi),
        "aggregate_match_seconds": sum(m.seconds for m in results),
    }


def tournament(games=250, seed=20261301, rollouts=1024, workers=4,
               csv_path=None, bootstrap=20000, training_dir=None):
    if games < 2 or games % 2:
        raise ValueError("Games must be positive and even")
    if workers < 1 or rollouts < 1 or bootstrap < 100:
        raise ValueError("Invalid tournament settings")
    started = time.perf_counter()
    matches = []
    if workers == 1:
        iterator = (play_match(i, seed, rollouts, training_dir) for i in range(games))
        for m in iterator:
            matches.append(m)
            print(f"MATCH {m.game} pair={m.pair} seed={m.deal_seed} "
                  f"seats=aware:{m.aware_seat},random:{m.random_policy_seat} "
                  f"scores=aware:{m.aware_score},random:{m.random_policy_score},"
                  f"third:{m.random_opponent_score} margin={m.margin:+d} "
                  f"winner={m.outright} seconds={m.seconds:.3f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for m in executor.map(play_match, range(games),
                                  repeat(seed), repeat(rollouts),
                                  repeat(training_dir), chunksize=1):
                matches.append(m)
                print(f"MATCH {m.game} pair={m.pair} seed={m.deal_seed} "
                      f"seats=aware:{m.aware_seat},random:{m.random_policy_seat} "
                      f"scores=aware:{m.aware_score},random:{m.random_policy_score},"
                      f"third:{m.random_opponent_score} margin={m.margin:+d} "
                      f"winner={m.outright} seconds={m.seconds:.3f}", flush=True)
    stats = summary(matches, seed=seed, bootstrap=bootstrap)
    stats["wall_seconds"] = time.perf_counter() - started
    print("SUMMARY " + " ".join(
        f"{k}={v}" for k, v in stats.items()
    ) + f" rollouts={rollouts} workers={workers} seed={seed}", flush=True)

    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(matches[0])))
            writer.writeheader()
            writer.writerows(asdict(m) for m in matches)
        print(f"CSV {path}", flush=True)
    return matches, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261301)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--training-dir", type=Path,
                        help="Optional per-game information-safe training traces")
    args = parser.parse_args()
    tournament(args.games, args.seed, args.rollouts, args.workers,
               args.csv, args.bootstrap, args.training_dir)


if __name__ == "__main__":
    main()
