"""Generate reproducible full 13-round Whist traces for neural training.

Each worker writes its own append-only JSONL. Match/round RNG resets make
initial deals independent of prior policy random-number consumption.

Examples:
 python -m benchmarks.generate_training_games --games 20 --agents random \
     --players 3 --output-dir data/trace-smoke
 python -m benchmarks.generate_training_games --games 2000 --workers 4 \
     --players 3 --agents random,mc_random,mc_aware --rollouts 256 \
     --output-dir data/mixed-traces

Use --private-dir only for privileged replay/debugging, never as model input.
"""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
from pathlib import Path
import random
import time

from alphawhist.core.WhistStructure import Game
from alphawhist.learning.recording import TrainingRecorder
from alphawhist.learning.timing import TimingCounters
from MonteCarloPlayer import MonteCarloPlayer
from RandomPlayer import RandomPlayer
from BidAware3Player import BidAware3Player

SCHEDULE = (7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7)
AGENTS = ("random", "aware3", "mc_random", "mc_aware")


def play_training_game(index, seed=20261102, players=3,
                       agents=("random",), rollouts=256,
                       output_dir="training-traces", private_dir=None,
                       timings_dir=None):
    if index < 0 or players < 2 or players > 7 or rollouts < 1:
        raise ValueError("Bad game index, player count or rollout budget")
    if not agents or any(a not in AGENTS for a in agents):
        raise ValueError("Unknown training agent")
    output_dir = Path(output_dir)
    trace = output_dir / f"game-{index:07d}.jsonl"
    if trace.exists():
        raise FileExistsError(f"Already recorded: {trace}")
    privileged = (Path(private_dir) / trace.name
                  if private_dir is not None else None)
    if privileged is not None and privileged.exists():
        raise FileExistsError(f"Already recorded: {privileged}")
    game_seed = seed + index * 1000003
    game = Game.__new__(Game)
    game.cardnumbers = list(SCHEDULE)
    game.playerConfigFileName = ""
    game.numberofplayers = players
    # Rotate agent types between positions and games, without introducing
    # card-count constraints that differ between agent types.
    names = [agents[(index + pid) % len(agents)] for pid in range(players)]
    game.players = []
    for pid, agent in enumerate(names):
        if agent == "random":
            player = RandomPlayer(pid)
        elif agent == "aware3":
            player = BidAware3Player(pid)
        else:
            params = {
                "monteCarloNumber": str(rollouts), "monteCarloWorkers": "1",
                "monteCarloPolicy":
                    "bid_aware_2" if agent == "mc_aware" else "random",
            }
            player = MonteCarloPlayer(pid, additionalParameters=params)
        game.players.append(player)
    game.resetGame()
    original_deal = game.randomDeal

    def fixed_deal(cards):
        random.seed(game_seed * 1009 + game.roundnumber)
        return original_deal(cards)

    game.randomDeal = fixed_deal
    timers = TimingCounters() if timings_dir is not None else None
    if timers is not None:
        for method in ("randomDeal", "getPlayerBids", "processScore", "observe"):
            timers.wrap(game, method, f"game.{method}")
        for pid, player in enumerate(game.players):
            timers.wrap(player, "makeMove", f"policy.{names[pid]}.makeMove")
    started = time.perf_counter()
    with TrainingRecorder(
        trace, game_id=f"{seed}-{index}", metadata={
            "run_seed": seed, "game_index": index, "game_seed": game_seed,
            "deal_group": f"generated-{seed}-{index}",
            "players": players, "agents": names,
            "rollout_budget": rollouts, "deal_algorithm": "seed_per_round_v1",
        }, privileged_path=privileged,
    ) as recorder:
        game.training_recorder = recorder
        if timers is not None:
            for method in ("on_round_start", "on_decision", "on_round_end",
                           "on_game_end"):
                timers.wrap(recorder, method, f"recording.{method}")
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                game.playFullGame()
        recorded = recorder.decisions_written
    elapsed = time.perf_counter() - started
    if game.roundnumber != len(SCHEDULE) or recorded != players * sum(SCHEDULE):
        raise AssertionError("Incomplete full-game training trace")
    if timers is not None:
        timers.save(
            Path(timings_dir) / f"game-{index:07d}.json",
            {"index": index, "agents": names, "players": players,
             "rollouts": rollouts, "total_wall_seconds": elapsed,
             "trace_bytes": trace.stat().st_size},
        )
    return index, recorded, elapsed


def generate(games=20, start=0, seed=20261102, players=3,
             agents=("random",), rollouts=256, workers=1,
             output_dir="training-traces", private_dir=None,
             timings_dir=None):
    if games < 1 or start < 0 or workers < 1:
        raise ValueError("Invalid game count, offset or workers")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = range(start, start + games)
    args = (seed, players, tuple(agents), rollouts, str(output_dir),
            str(private_dir) if private_dir else None,
            str(timings_dir) if timings_dir else None)
    started = time.perf_counter()
    if workers == 1:
        iterator = (play_training_game(index, *args) for index in indices)
        for index, count, seconds in iterator:
            print(f"GAME {index} decisions={count} seconds={seconds:.2f}",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, count, seconds in pool.map(
                    play_training_game, indices,
                    repeat(seed), repeat(players), repeat(tuple(agents)),
                    repeat(rollouts), repeat(str(output_dir)),
                    repeat(str(private_dir) if private_dir else None),
                    repeat(str(timings_dir) if timings_dir else None),
                    chunksize=1):
                print(f"GAME {index} decisions={count} seconds={seconds:.2f}",
                      flush=True)
    print(f"DONE games={games} players={players} "
          f"decisions={games * players * sum(SCHEDULE)} "
          f"wall_seconds={time.perf_counter()-started:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--agents", default="random",
                        help="Comma-separated random,aware3,mc_random,mc_aware")
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--timings-dir", type=Path,
                        help="Opt-in per-game JSON breakdown of recording, observation and policy time")
    args = parser.parse_args()
    generate(args.games, args.start, args.seed, args.players,
             tuple(args.agents.split(",")), args.rollouts,
             args.workers, args.output_dir, args.private_dir,
             args.timings_dir)


if __name__ == "__main__":
    main()
