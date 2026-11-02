"""Profile *only* neural-guided MC card selection in a real Whist game.

The usual tournament uses separate processes, so wrapping the parent CLI in
cProfile will NOT capture the hot work inside the four game workers. This
script runs one reproducible 3-player match in a single process and enables
cProfile only for non-forced NeuralRolloutMonteCarloPlayer.makeMove() calls.

It writes:
  <prefix>.prof   raw cProfile data (view with pstats or snakeviz)
  <prefix>.txt    cumulative-time and self-time hotspots
  <prefix>.json   machine-readable timings, counts and grouped hot functions

Example:
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
 python -m benchmarks.profile_neural_rollout \\
   --checkpoint runs/large-190k-v1/best.pt \\
   --rollouts 32 --rounds 3 --seed 90061101 \\
   --output-prefix runs/large-190k-v1/profile-neural-mc-32

cProfile slows down Python execution, so profile timings should be used to
identify bottlenecks, NOT as an uninstrumented gameplay throughput measurement.
"""
import argparse
import contextlib
import cProfile
import json
import os
from pathlib import Path
import pstats
import random
import time

from alphawhist.core.WhistStructure import Game
from NeuralRolloutMonteCarloPlayer import NeuralRolloutMonteCarloPlayer
from NeuralValuePlayer import NeuralValuePlayer
from RandomPlayer import RandomPlayer


# Display cumulative times for named functions, without summing the numbers:
# a caller's cumulative time already includes its children.
_HOTSPOTS = (
    ("rollout all branches", "neural_rollout.py", "_neural_batch"),
    ("advance active branches", "neural_rollout.py", "_finish_group"),
    ("opponent / forced play", "neural_rollout.py", "_until_neural_turn"),
    ("advance public observation", "neural_rollout.py", "_advance"),
    ("public event reducer", "InformationState.py", "apply_public_event"),
    ("observation validation", "InformationState.py", "__post_init__"),
    ("feature batch", "encoding.py", "encode_batch"),
    ("encode one observation", "encoding.py", "encode_decision"),
    ("neural inference", "model.py", "predict_encoded"),
    ("sample hidden hands", "InformationState.py", "sample_hands"),
    ("simulate one legal card", "FastRollout.py", "_play_legal"),
    ("opponent policy choice", "FastRollout.py", "_choose_card"),
)


def grouped_hotspots(profile):
    """Return non-additive, cumulative-time stats for specific call sites."""
    results = []
    for label, filename, funcname in _HOTSPOTS:
        matches = [
            (primitive, total, internal, cumulative)
            for (path, _line, name), (primitive, total, internal,
                                      cumulative, _callers) in
            pstats.Stats(profile).stats.items()
            if Path(path).name == filename and name == funcname
        ]
        results.append({
            "stage": label,
            "calls": sum(total for _, total, _, _ in matches),
            "primitive_calls": sum(p for p, _, _, _ in matches),
            "self_seconds": round(sum(t for _, _, t, _ in matches), 6),
            "cumulative_seconds": round(sum(t for _, _, _, t in matches), 6),
        })
    return results


def profile_game(checkpoint, *, rollouts=32, rounds=3, seed=90061101,
                 output_prefix="runs/profile-neural-mc-32",
                 worlds_per_batch=16):
    """Run reproducible real rounds while profiling search decisions only."""
    if not 1 <= rounds <= 13:
        raise ValueError("rounds must be between 1 and 13")
    if rollouts < 1 or worlds_per_batch < 1:
        raise ValueError("rollouts and worlds_per_batch must be positive")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Load outside the timed section so weight deserialization doesn't obscure
    # the per-decision steady-state execution we want to optimize.
    from alphawhist.learning.neural_rollout import _load_model
    _load_model(str(checkpoint))
    random.seed(seed)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = game.cardnumbers[:rounds]
    challenger = NeuralRolloutMonteCarloPlayer(
        0, checkpoint=str(checkpoint), rollouts=rollouts,
        worlds_per_batch=worlds_per_batch,
    )
    game.players[0] = challenger
    game.players[1] = NeuralValuePlayer(1, checkpoint=str(checkpoint))
    assert type(game.players[2]) is RandomPlayer

    # Same deterministic deal schedule used by the head-to-head tournament.
    original_deal = game.randomDeal

    def seeded_deal(ncards):
        random.seed(seed * 1009 + game.roundnumber)
        return original_deal(ncards)

    game.randomDeal = seeded_deal
    profile = cProfile.Profile()
    original_move = challenger.makeMove
    decisions = 0
    nonforced = 0
    profiled_search_wall = 0.0

    def measured_move(pile, trumpsuit, fullGameObject=None):
        nonlocal decisions, nonforced, profiled_search_wall
        decisions += 1
        # This mirror of the legal-card check is deliberately OUTSIDE the
        # profiler; forced moves have no neural search to optimize.
        legal = challenger.getRealPossibleMoves(pile)
        if len(legal) <= 1:
            return original_move(pile, trumpsuit, fullGameObject)
        nonforced += 1
        started = time.perf_counter()
        profile.enable()
        try:
            return original_move(pile, trumpsuit, fullGameObject)
        finally:
            profile.disable()
            profiled_search_wall += time.perf_counter() - started

    challenger.makeMove = measured_move
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            game.playFullGame()
    game_seconds = time.perf_counter() - started

    profile_path = Path(str(prefix) + ".prof")
    text_path = Path(str(prefix) + ".txt")
    json_path = Path(str(prefix) + ".json")
    profile.dump_stats(str(profile_path))
    with text_path.open("w", encoding="utf-8") as stream:
        print("CUMULATIVE TIME (includes callees)\n", file=stream)
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
            pstats.SortKey.CUMULATIVE).print_stats(75)
        print("\nSELF TIME (excluding callees)\n", file=stream)
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
            pstats.SortKey.TIME).print_stats(75)
    report = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "rollouts_per_candidate": rollouts,
        "rounds": rounds,
        "worlds_per_batch": worlds_per_batch,
        "player": "neural_guided_monte_carlo",
        "baseline": "direct_neural",
        "game_wall_seconds_under_profiler": round(game_seconds, 6),
        "search_wall_seconds_under_profiler": round(profiled_search_wall, 6),
        "challenger_decisions": decisions,
        "challenger_nonforced_searches": nonforced,
        "challenger_score": game.scores[0],
        "baseline_score": game.scores[1],
        "random_score": game.scores[2],
        "hotspots": grouped_hotspots(profile),
        "note": "Hotspot cumulative times overlap and are not additive. "
                "cProfile can substantially inflate wall time; compare "
                "uninstrumented tournament wall time separately.",
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"PROFILE_DONE rounds={rounds} decisions={decisions} "
          f"nonforced_searches={nonforced} "
          f"search_seconds_profiled={profiled_search_wall:.2f} "
          f"game_seconds_profiled={game_seconds:.2f}", flush=True)
    print("HOTSPOTS (cumulative seconds; nested values overlap)", flush=True)
    for entry in sorted(report["hotspots"],
                        key=lambda row: row["cumulative_seconds"],
                        reverse=True):
        if entry["calls"]:
            print(f" {entry['cumulative_seconds']:8.3f}s "
                  f"({entry['self_seconds']:7.3f}s self) "
                  f"{entry['calls']:>8} calls  {entry['stage']}", flush=True)
    print(f"PROFILES {profile_path} {text_path} {json_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of real 13-round match rounds (default 3)")
    parser.add_argument("--seed", type=int, default=90061101)
    parser.add_argument("--worlds-per-batch", type=int, default=16)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("runs/profile-neural-mc-32"))
    args = parser.parse_args()
    profile_game(
        args.checkpoint, rollouts=args.rollouts, rounds=args.rounds,
        seed=args.seed, output_prefix=args.output_prefix,
        worlds_per_batch=args.worlds_per_batch,
    )


if __name__ == "__main__":
    main()
