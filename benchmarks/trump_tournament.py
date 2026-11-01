"""Compare expected-score and legacy heuristic trump pickers head-to-head.

Both competitors use the SAME MonteCarloPlayer bidding and 1024-rollout
card-playing policy; only pick_trumps differs. A RandomPlayer fills the
third seat. 50 matches = 25 paired deals, each played twice with the two
Monte Carlo agents swapping seats. Pairs rotate which two seats compete.

Run:
    python -m benchmarks.trump_tournament --games 50 --rollouts 1024
    python -m benchmarks.trump_tournament --games 50 --rollouts 1024 --csv tournament.csv
    python -m benchmarks.trump_tournament --games 500 --rollouts 1024 --workers 4 --csv tournament.csv
    python -m benchmarks.trump_tournament --games 250 --rollouts 1024 --workers 4 --seed 20261056 --trumps-csv trump-choices.csv

The two matches in a pair have identical starting deals in every round,
but decisions and random streams can diverge as the policies interact.
Use --workers N to evaluate independent matches concurrently in separate
processes; each agent still uses one worker for its 1024 rollouts per move.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import contextlib
import csv
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import statistics
import time

from alphawhist.core import TrumpPicker
from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from RandomPlayer import RandomPlayer

PAIR_SEATS = ((0, 1), (0, 2), (1, 2))


class HeuristicTrumpMonteCarloPlayer(MonteCarloPlayer):
    """Unchanged card/bid intelligence, with Ed Dakin's original trump choice."""

    def pick_trumps(self, nplayers):
        return TrumpPicker.picktrump(self.real_hand)


@dataclass(frozen=True)
class Match:
    game: int
    deal_seed: int
    new_seat: int
    old_seat: int
    random_seat: int
    new_score: int
    old_score: int
    random_score: int
    margin: int
    result: str
    winner: str
    seconds: float


def play_match(game_index, seed=20261031, rollouts=1024, include_trumps=False):
    if game_index < 0 or rollouts < 1:
        raise ValueError("game_index must be >= 0 and rollouts must be positive")
    pair = game_index // 2
    left, right = PAIR_SEATS[pair % len(PAIR_SEATS)]
    new_seat, old_seat = (left, right) if game_index % 2 == 0 else (right, left)
    random_seat = ({0, 1, 2} - {new_seat, old_seat}).pop()
    deal_seed = seed + pair  # paired seat-swap uses exactly the same deals
    random.seed(deal_seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    params = {"monteCarloNumber": str(rollouts), "monteCarloWorkers": "1"}
    game.players[new_seat] = MonteCarloPlayer(new_seat, additionalParameters=params)
    game.players[old_seat] = HeuristicTrumpMonteCarloPlayer(
        old_seat, additionalParameters=params
    )
    assert isinstance(game.players[random_seat], RandomPlayer)

    # Reset random dealing per round independently of how much RNG was used
    # by a picker, a bidder, or any policy in an earlier round.
    ordinary_deal = game.randomDeal

    def fixed_deal(cards_in_round):
        random.seed(deal_seed * 1009 + game.roundnumber)
        return ordinary_deal(cards_in_round)

    game.randomDeal = fixed_deal
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    duration = time.perf_counter() - started

    assert game.roundnumber == 13
    history = game.observe(0).public_history
    assert sum(e["type"] == "round_end" for e in history) == 13
    assert sum(e["type"] == "trick_end" for e in history) == 55
    assert sum(e["type"] == "card" for e in history) == 165

    new_score = game.scores[new_seat]
    old_score = game.scores[old_seat]
    random_score = game.scores[random_seat]
    margin = new_score - old_score
    result = "NEW" if margin > 0 else "OLD" if margin < 0 else "TIE"
    max_score = max(game.scores)
    winners = [i for i, score in enumerate(game.scores) if score == max_score]
    winner = (
        "NEW" if winners == [new_seat] else
        "OLD" if winners == [old_seat] else
        "RANDOM" if winners == [random_seat] else "TIE_FOR_FIRST"
    )
    match = Match(
        game=game_index + 1, deal_seed=deal_seed,
        new_seat=new_seat, old_seat=old_seat, random_seat=random_seat,
        new_score=new_score, old_score=old_score, random_score=random_score,
        margin=margin, result=result, winner=winner, seconds=duration
    )
    if not include_trumps:
        return match
    # Public history records all 13 trump selections. Recording them does not
    # cause any additional random draws or alter gameplay, unlike recomputing
    # the Monte Carlo picker's choices after the fact.
    trump_events = [
        (round_index, event["player"], event["suit"])
        for round_index, event in enumerate(
            event for event in history if event["type"] == "trump"
        )
    ]
    assert len(trump_events) == 13
    assert all(dealer == index % 3
               for index, dealer, suit in trump_events)
    return match, tuple(trump_events)


def tournament(games=50, seed=20261031, rollouts=1024, csv_path=None,
               workers=1, trumps_csv=None):
    if games < 1 or games % 2:
        raise ValueError("Number of games must be positive and even (seat-swap pairs)")
    if rollouts < 1 or workers < 1:
        raise ValueError("Rollouts and workers must be positive")
    results = []
    trump_rows = []
    started = time.perf_counter()

    def record(item):
        if trumps_csv is not None:
            match, picks = item
            for round_index, dealer, suit in picks:
                picker = (
                    "NEW" if dealer == match.new_seat else
                    "OLD" if dealer == match.old_seat else "RANDOM"
                )
                trump_rows.append({
                    "game": match.game,
                    "pair": (match.game + 1) // 2,
                    "deal_seed": match.deal_seed,
                    "round": round_index + 1,
                    "cards": (7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7)[round_index],
                    "dealer": dealer,
                    "picker": picker,
                    "suit": suit,
                })
        else:
            match = item
        results.append(match)
        print(
            f"MATCH {match.game:02d} pair={((match.game - 1) // 2) + 1:02d} "
            f"deal_seed={match.deal_seed} seats="
            f"new:{match.new_seat},old:{match.old_seat},random:{match.random_seat} "
            f"scores=new:{match.new_score},old:{match.old_score},random:{match.random_score} "
            f"margin={match.margin:+d} h2h={match.result} "
            f"outright={match.winner} time={match.seconds:.3f}s",
            flush=True
        )

    if workers == 1:
        for index in range(games):
            record(play_match(index, seed=seed, rollouts=rollouts,
                              include_trumps=trumps_csv is not None))
    else:
        # Ordered results despite concurrent execution; games seed their
        # process-local global RNG independently, so outcomes remain exact.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for match in pool.map(play_match, range(games),
                                  repeat(seed), repeat(rollouts),
                                  repeat(trumps_csv is not None), chunksize=1):
                record(match)
    elapsed = time.perf_counter() - started
    new_h2h = sum(m.result == "NEW" for m in results)
    old_h2h = sum(m.result == "OLD" for m in results)
    ties = sum(m.result == "TIE" for m in results)
    assert new_h2h + old_h2h + ties == games
    new_out = sum(m.winner == "NEW" for m in results)
    old_out = sum(m.winner == "OLD" for m in results)
    random_out = sum(m.winner == "RANDOM" for m in results)
    top_ties = sum(m.winner == "TIE_FOR_FIRST" for m in results)
    print(
        f"SUMMARY games={games} paired_deals={games // 2} "
        f"rollouts={rollouts} workers={workers} h2h_new_wins={new_h2h} "
        f"h2h_old_wins={old_h2h} h2h_ties={ties} "
        f"outright_new_wins={new_out} outright_old_wins={old_out} "
        f"outright_random_wins={random_out} outright_ties={top_ties} "
        f"avg_new_score={statistics.mean(m.new_score for m in results):.3f} "
        f"avg_old_score={statistics.mean(m.old_score for m in results):.3f} "
        f"avg_random_score={statistics.mean(m.random_score for m in results):.3f} "
        f"avg_margin={statistics.mean(m.margin for m in results):+.3f} "
        f"median_margin={statistics.median(m.margin for m in results):+.3f} "
        f"total_seconds={sum(m.seconds for m in results):.3f} "
        f"wall_seconds={elapsed:.3f}",
        flush=True
    )
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            writer.writerows(asdict(match) for match in results)
        print(f"CSV {csv_path}", flush=True)
    if trumps_csv is not None:
        trump_path = Path(trumps_csv)
        trump_path.parent.mkdir(parents=True, exist_ok=True)
        assert len(trump_rows) == games * 13
        with trump_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("game", "pair", "deal_seed", "round", "cards",
                            "dealer", "picker", "suit"),
            )
            writer.writeheader()
            writer.writerows(trump_rows)
        print(f"TRUMP_CSV {trump_path} selections={len(trump_rows)}",
              flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent full-game processes, distinct from rollout workers")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--trumps-csv", type=Path,
                        help="Export every actually chosen trump suit, by "
                             "dealer and game, from public game history")
    args = parser.parse_args()
    tournament(args.games, args.seed, args.rollouts, args.csv,
               workers=args.workers, trumps_csv=args.trumps_csv)


if __name__ == "__main__":
    main()
