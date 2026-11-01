"""Run full Whist matches against two random opponents and count winners.

Usage (from the repository root):
  python -m benchmarks.ten_games_vs_random
  python -m benchmarks.ten_games_vs_random --games 100 --rollouts 1024

For a fairer comparison, both AI variants receive the same initial *deals*
for each match seed/seat: the shuffle RNG is reseeded before every round.
Opponent random moves can diverge as games evolve, so this is not a
common-random-actions experiment. A tie for highest score is not a win.
"""
import argparse
import contextlib
from dataclasses import dataclass
import os
import random
import statistics
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer

BOTS = {
    "MonteCarloPlayer": MonteCarloPlayer,
    "SnakeMonteCarloPlayer": SnakeMonteCarloPlayer,
}
DEFAULT_GAMES = 10
DEFAULT_SEED = 20261031


@dataclass(frozen=True)
class MatchResult:
    bot: str
    game: int
    seed: int
    seat: int
    scores: tuple[int, ...]
    outcome: str
    seconds: float

    @property
    def ai_score(self):
        return self.scores[self.seat]

    @property
    def rival_scores(self):
        return tuple(score for i, score in enumerate(self.scores) if i != self.seat)

    @property
    def margin(self):
        return self.ai_score - max(self.rival_scores)


def play_match(bot_class, seed, seat, rollouts=1024, workers=1, game_number=1):
    if seat not in (0, 1, 2):
        raise ValueError("The game has three player seats")
    if rollouts < 1 or workers < 1:
        raise ValueError("Rollouts and workers must be positive")

    random.seed(seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.players[seat] = bot_class(
        seat, additionalParameters={
            "monteCarloNumber": str(rollouts),
            "monteCarloWorkers": str(workers),
        }
    )
    # Before each round, use a predetermined shuffle, unaffected by how
    # many RNG draws either AI consumed in previous rounds.
    deal = game.randomDeal

    def fixed_round_deal(cards_in_round):
        random.seed(seed * 1009 + game.roundnumber)
        return deal(cards_in_round)

    game.randomDeal = fixed_round_deal

    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullGame()
    elapsed = time.perf_counter() - started

    assert game.roundnumber == 13
    assert game.observe(0).phase == "between_rounds"
    assert sum(e["type"] == "trick_end" for e in game.observe(0).public_history) == 55

    scores = tuple(game.scores)
    leaders = [i for i, s in enumerate(scores) if s == max(scores)]
    outcome = "WIN" if leaders == [seat] else ("TIE" if seat in leaders else "LOSS")
    return MatchResult(
        bot_class.__name__, game_number, seed, seat, scores, outcome, elapsed
    )


def run_experiment(games=DEFAULT_GAMES, rollouts=1024, workers=1,
                   seed=DEFAULT_SEED, bots=tuple(BOTS)):
    if games < 1:
        raise ValueError("At least one game is required")
    if any(name not in BOTS for name in bots):
        raise ValueError("Unknown AI bot")
    output = {}
    for name in bots:
        matches = []
        print(f"\nBOT {name}, games={games}, rollouts={rollouts}, workers={workers}",
              flush=True)
        for i in range(games):
            seat = i % 3
            result = play_match(
                BOTS[name], seed=seed + i, seat=seat,
                rollouts=rollouts, workers=workers, game_number=i + 1
            )
            matches.append(result)
            print(
                f"GAME {i + 1:2d} seat={seat} seed={result.seed} "
                f"scores={list(result.scores)} ai={result.ai_score} "
                f"opponents={list(result.rival_scores)} "
                f"margin={result.margin:+d} outcome={result.outcome} "
                f"time={result.seconds:.3f}s",
                flush=True,
            )

        wins = sum(x.outcome == "WIN" for x in matches)
        ties = sum(x.outcome == "TIE" for x in matches)
        losses = sum(x.outcome == "LOSS" for x in matches)
        assert wins + ties + losses == games
        print(
            f"SUMMARY bot={name} wins={wins} ties={ties} losses={losses} "
            f"games={games} outright_win_rate={wins / games:.1%} "
            f"avg_ai_score={statistics.mean(x.ai_score for x in matches):.2f} "
            f"avg_rival_best={statistics.mean(max(x.rival_scores) for x in matches):.2f} "
            f"avg_margin={statistics.mean(x.margin for x in matches):+.2f} "
            f"total_time={sum(x.seconds for x in matches):.3f}s",
            flush=True
        )
        output[name] = matches
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bots", choices=BOTS, nargs="+",
                        default=list(BOTS))
    args = parser.parse_args()
    run_experiment(
        games=args.games, rollouts=args.rollouts, workers=args.workers,
        seed=args.seed, bots=args.bots
    )


if __name__ == "__main__":
    main()
