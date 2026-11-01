"""Matched single-round A/B test for Monte Carlo continuation policies.

Both candidate bots play the SAME seat in independent replays of the SAME
dealt round, against fixed, independently seeded opponents (either random
or strong 1024-rollout Monte Carlo bots).
Trump selection, bids, and starting hands are checked for equality. Each
candidate has the same rollout budget; only monteCarloPolicy differs.

All scores are actual authoritative Game round scores, NOT simulation estimates.
Scenario seeds identify independent deals. No real opponent hands are given to
the acting MonteCarloPlayer; it observes only public history and its own hand.

Run:
 python -m benchmarks.paired_round_policy --rounds-per-size 100 --cards 1 2 3 4 5 6 7 \
     --rollouts 1024 --workers 4 --opponents monte_carlo --csv paired-rounds.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, redirect_stdout
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

from alphawhist.core.WhistLib import all_cards_in_deck, tricksToScore
from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from RandomPlayer import RandomPlayer

POLICIES = ("random", "bid_aware_2")


class _SeededOpponent(RandomPlayer):
    """Same isolated opponent RNG stream in both A/B replays."""

    def __init__(self, player_number, seed):
        super().__init__(player_number)
        self._rng = random.Random(seed)

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        choices = self.getRealPossibleMoves(pile)
        card = self._rng.choice(choices)
        if not self.playCard(card):
            raise ValueError("Opponent chose a card it did not hold")
        self.cardSuitCheck(pile, card)
        return card


class _SeededMonteCarlo(MonteCarloPlayer):
    """Stable independent rollout seed for each public decision position."""

    def __init__(self, player_number, rollouts, policy, seed):
        super().__init__(player_number, additionalParameters={
            "monteCarloNumber": str(rollouts),
            "monteCarloWorkers": "1",
            "monteCarloPolicy": policy,
        })
        self._seed = seed

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        if fullGameObject is None:
            raise ValueError("A Game is required")
        # Neither opponents' cards nor how many random numbers they drew
        # affects this move's possible-world sampling seed.
        random.seed(self._seed + 1000003 * fullGameObject.trickNumber
                    + 1009 * len(pile))
        return super().makeMove(pile, trumpsuit, fullGameObject)


@dataclass(frozen=True)
class RoundResult:
    scenario: int
    cards: int
    players: int
    opponents: str
    deal_seed: int
    deal_sha256: str
    tested_seat: int
    dealer: int
    trump: str
    bids: str
    policy_a: str
    policy_b: str
    score_a: int
    score_b: int
    score_delta: int
    margin_a: int
    margin_b: int
    margin_delta: int
    tricks_a: int
    tricks_b: int
    exact_bid_a: int
    exact_bid_b: int
    seconds: float


def _play_round(hands, dealer, seat, rollouts, policy, seed,
                opponents="random"):
    nplayers = len(hands)
    cards = len(hands[0])
    game = Game.__new__(Game)
    game.cardnumbers = [cards]
    game.playerConfigFileName = ""
    game.numberofplayers = nplayers
    if opponents not in ("random", "monte_carlo"):
        raise ValueError("Unknown opponent type")
    game.players = [
        (_SeededMonteCarlo(pid, rollouts, policy, seed ^ 0x43A125)
         if pid == seat else
         _SeededMonteCarlo(pid, rollouts, "bid_aware_2",
                           (seed ^ 0x53B123) + 20011 * (pid + 1))
         if opponents == "monte_carlo" else
         _SeededOpponent(pid, seed + 20011 * (pid + 1)))
        for pid in range(nplayers)
    ]
    game.resetGame()
    game.dealer = dealer
    game.leader = (dealer + 1) % nplayers

    # Replicate Game.randomDeal's information boundary: player zero knows
    # its own hand, not opponents' real hands. Each player is later observed
    # from its own private hand and public events.
    def fixed_deal(requested):
        if requested != cards:
            raise ValueError("Incorrect number of cards requested")
        unseen_from_zero = [c for c in all_cards_in_deck if c not in hands[0]]
        for pid, player in enumerate(game.players):
            player.set_hands(hands[pid], hands[pid] if pid == 0
                             else unseen_from_zero)
        game._begin_observer_round(cards)

    game.randomDeal = fixed_deal
    random.seed(seed)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink):
            game.playFullRound()
    history = game.observe(0).public_history
    assert game.roundnumber == 1
    assert sum(event["type"] == "card" for event in history) == cards * nplayers
    assert sum(event["type"] == "trick_end" for event in history) == cards
    assert sum(game.tricksWon) == cards
    assert all(len(p.real_hand) == 0 for p in game.players)
    assert game.scores == [
        tricksToScore(bid, won) for bid, won in zip(game.bids, game.tricksWon)
    ]
    return (tuple(game.scores), tuple(game.tricksWon), tuple(game.bids),
            game.trumpsuit)


def run_scenario(index, cards, seed=20261102, rollouts=256,
                 players=3, policy_a="random", policy_b="bid_aware_2",
                 opponents="random"):
    """One independent, controlled matched-deal pair of actual round replays."""
    if index < 0 or rollouts < 1 or players < 2 or players > 7:
        raise ValueError("Invalid index, rollout budget or player count")
    if cards < 1 or cards > 7 or cards * players > 52:
        raise ValueError("Invalid cards per player")
    if policy_a not in POLICIES or policy_b not in POLICIES:
        raise ValueError("Unsupported continuation policy")
    if opponents not in ("random", "monte_carlo"):
        raise ValueError("Unknown opponent type")
    deal_seed = seed + cards * 1000003 + index
    rng = random.Random(deal_seed)
    dealt = rng.sample(all_cards_in_deck, cards * players)
    hands = tuple(
        tuple(dealt[pid * cards:(pid + 1) * cards])
        for pid in range(players)
    )
    dealer = rng.randrange(players)
    seat = index % players
    fingerprint = hashlib.sha256(json.dumps(hands).encode()).hexdigest()
    begin = time.perf_counter()
    a = _play_round(hands, dealer, seat, rollouts, policy_a, deal_seed,
                    opponents)
    b = _play_round(hands, dealer, seat, rollouts, policy_b, deal_seed,
                    opponents)
    scores_a, tricks_a, bids_a, trump_a = a
    scores_b, tricks_b, bids_b, trump_b = b
    # Only the continuation policy is changed. Both bids and trump are
    # fixed by identical deterministic pre-play logic, not by the rollout.
    if bids_a != bids_b or trump_a != trump_b:
        raise AssertionError("Matched round changed its initial bids or trump")
    margin_a = scores_a[seat] - max(x for i, x in enumerate(scores_a)
                                     if i != seat)
    margin_b = scores_b[seat] - max(x for i, x in enumerate(scores_b)
                                     if i != seat)
    return RoundResult(
        scenario=index, cards=cards, players=players, opponents=opponents,
        deal_seed=deal_seed,
        deal_sha256=fingerprint, tested_seat=seat, dealer=dealer,
        trump=trump_a, bids=json.dumps(bids_a),
        policy_a=policy_a, policy_b=policy_b,
        score_a=scores_a[seat], score_b=scores_b[seat],
        score_delta=scores_b[seat] - scores_a[seat],
        margin_a=margin_a, margin_b=margin_b, margin_delta=margin_b - margin_a,
        tricks_a=tricks_a[seat], tricks_b=tricks_b[seat],
        exact_bid_a=int(bids_a[seat] == tricks_a[seat]),
        exact_bid_b=int(bids_b[seat] == tricks_b[seat]),
        seconds=time.perf_counter() - begin,
    )


def _run_job(args):
    return run_scenario(*args)


def bootstrap_ci(values, seed=0, iterations=4000, alpha=0.01):
    """Percentile bootstrap over independently seeded matched round pairs."""
    if not values or iterations < 100 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap arguments")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(iterations)
    )
    return (means[int(iterations * alpha / 2)],
            means[min(iterations - 1, int(iterations * (1 - alpha / 2)))])


def summarize(results, bootstrap=4000, seed=20261102):
    if not results:
        raise ValueError("Need completed matched rounds")
    grouped = [(0, results)] + [
        (cards, [r for r in results if r.cards == cards])
        for cards in sorted(set(r.cards for r in results))
    ]
    stats = []
    for cards, matches in grouped:
        deltas = [r.score_delta for r in matches]
        lo, hi = bootstrap_ci(deltas, seed=seed + cards, iterations=bootstrap)
        stats.append({
            "cards": cards, "rounds": len(matches),
            "mean_delta": statistics.mean(deltas),
            "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.,
            "ci99": (lo, hi),
            "wins": sum(d > 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
            "exact_delta": statistics.mean(r.exact_bid_b - r.exact_bid_a
                                           for r in matches),
            "mean_margin_delta": statistics.mean(r.margin_delta for r in matches),
        })
    return stats


def tournament(rounds_per_size=100, cards=(1, 2, 3, 4, 5, 6, 7),
               rollouts=256, workers=4, players=3, seed=20261102,
               csv_path=None, bootstrap=4000,
               policy_a="random", policy_b="bid_aware_2",
               progress_every=25, opponents="random"):
    if rounds_per_size < 1 or workers < 1 or bootstrap < 100 or progress_every < 1:
        raise ValueError("Invalid tournament settings")
    cards = tuple(cards)
    if len(cards) != len(set(cards)) or not cards:
        raise ValueError("Card counts must be distinct and nonempty")
    if any(not 1 <= c <= 7 or c * players > 52 for c in cards):
        raise ValueError("Invalid card size for player count")
    if opponents not in ("random", "monte_carlo"):
        raise ValueError("Unknown opponent type")
    jobs = [(i, c, seed, rollouts, players, policy_a, policy_b, opponents)
            for c in cards for i in range(rounds_per_size)]
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = []
    with (open(csv_path, "w", encoding="utf-8", newline="")
          if csv_path is not None else nullcontext()) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RoundResult.__dataclass_fields__)) \
            if stream is not None else None
        if writer:
            writer.writeheader()
            stream.flush()
        with (ProcessPoolExecutor(max_workers=workers)
              if workers > 1 else nullcontext()) as pool:
            iterator = (pool.map(_run_job, jobs, chunksize=1)
                        if pool is not None else map(_run_job, jobs))
            for result in iterator:
                completed.append(result)
                if writer:
                    writer.writerow(asdict(result))
                    stream.flush()  # Survives a CI timeout or cancellation.
                if len(completed) % progress_every == 0 or len(completed) == len(jobs):
                    deltas = [r.score_delta for r in completed]
                    print(
                        f"PROGRESS {len(completed)}/{len(jobs)} "
                        f"mean_score_delta={statistics.mean(deltas):+.3f} "
                        f"B_wins={sum(x > 0 for x in deltas)} "
                        f"A_wins={sum(x < 0 for x in deltas)} "
                        f"ties={sum(x == 0 for x in deltas)}",
                        flush=True,
                    )
    stats = summarize(completed, bootstrap=bootstrap, seed=seed)
    for item in stats:
        print("SUMMARY " + " ".join(f"{key}={value}"
                                  for key, value in item.items()), flush=True)
    print(f"DONE rounds={len(completed)} opponents={opponents} "
          f"policies={policy_a},{policy_b} "
          f"rollouts={rollouts} workers={workers} seed={seed} "
          f"wall_seconds={time.perf_counter() - started:.3f}", flush=True)
    return completed, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-per-size", type=int, default=100)
    parser.add_argument("--cards", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--opponents", choices=("random", "monte_carlo"),
                        default="random")
    parser.add_argument("--policy-a", choices=POLICIES, default="random")
    parser.add_argument("--policy-b", choices=POLICIES, default="bid_aware_2")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    tournament(args.rounds_per_size, args.cards, args.rollouts,
               args.workers, args.players, args.seed, args.csv, args.bootstrap,
               args.policy_a, args.policy_b, args.progress_every,
               args.opponents)


if __name__ == "__main__":
    main()
