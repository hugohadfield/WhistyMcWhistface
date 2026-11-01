"""Profile a complete 13-round Whist game with a Monte Carlo agent.

Includes dealing, trump selection, bidding, all 165 card plays, scoring,
public-observation maintenance and 1024 rollouts at nonforced AI decisions.

Run from repository root:
    python -m benchmarks.profile_full_game
    python -m benchmarks.profile_full_game --bot SnakeMonteCarloPlayer --limit 50
    python -m benchmarks.profile_full_game --rollouts 1024 --output match.prof

Only one worker is supported: cProfile in the parent cannot see child
ProcessPoolExecutor CPU time. The wall time with cProfile is *instrumented*
and will be higher than an ordinary game.
"""
import argparse
import cProfile
import contextlib
import io
import os
from pathlib import Path
import pstats
import random
import time

from alphawhist.core.WhistStructure import Game
from MonteCarloPlayer import MonteCarloPlayer
from SnakeMonteCarloPlayer import SnakeMonteCarloPlayer

BOT_CLASSES = {
    "MonteCarloPlayer": MonteCarloPlayer,
    "SnakeMonteCarloPlayer": SnakeMonteCarloPlayer,
}


def play_profiled_game(bot_name="MonteCarloPlayer", rollouts=1024,
                       seed=20261031, seat=0, limit=35, output=None):
    """Run and print cProfile cumulative and exclusive CPU breakdowns.

    Returns a (scores, elapsed_seconds, stats) tuple. The full match and
    player setup are within the profiler region; stdout for gameplay only is
    sent to os.devnull, not retained in a large buffer.
    """
    if bot_name not in BOT_CLASSES:
        raise ValueError("Unknown AI bot")
    if rollouts < 1 or seat not in (0, 1, 2) or limit < 1:
        raise ValueError("Invalid rollouts, seat or limit")

    def game_workload():
        random.seed(seed)
        game = Game("testConfigFiles/ThreePlayerRandom.json")
        game.players[seat] = BOT_CLASSES[bot_name](
            seat,
            additionalParameters={
                "monteCarloNumber": str(rollouts),
                "monteCarloWorkers": "1",
            },
        )
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                game.playFullGame()
        assert game.roundnumber == 13
        assert game.observe(0).phase == "between_rounds"
        return tuple(game.scores)

    profiler = cProfile.Profile()
    started = time.perf_counter()
    scores = profiler.runcall(game_workload)
    elapsed = time.perf_counter() - started

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(path))
        print(f"PROFILE_FILE {path}")

    stats = pstats.Stats(profiler)
    stats.total_tt = sum(item[2] for item in stats.stats.values())
    print(
        f"PROFILE_MATCH bot={bot_name} rollouts={rollouts} seed={seed} "
        f"seat={seat} workers=1 rounds=13 tricks=55 card_plays=165 "
        f"wall_seconds={elapsed:.3f} profile_cpu_seconds={stats.total_tt:.3f} "
        f"scores={scores}",
        flush=True
    )

    def summarize(label, match):
        # pstats cumulative time is inclusive of callees and should not be
        # summed across nested functions. Report one line per function.
        rows = [
            (function, data)
            for function, data in stats.stats.items()
            if match(function)
        ]
        rows.sort(key=lambda x: x[1][3], reverse=True)
        for (filename, line, function), (primitive, calls, own, cumulative, _) in rows:
            print(
                f"HOTSPOT group={label} fn={function} file={Path(filename).name} "
                f"line={line} calls={calls} self_s={own:.4f} cum_s={cumulative:.4f} "
                f"self_pct={100 * own / stats.total_tt:.1f}%"
            )

    print("KEY_HOTSPOTS (cumulative time is inclusive and must not be added):")
    interesting = {
        "simulation": {"evaluate_actions", "_batch", "finish_round", "play",
                       "from_snapshot", "sample_hands", "_choose_card",
                       "legal_mask", "trick_winner", "round_scores"},
        "observation": {"observe", "apply_public_event", "_record_public_event",
                        "__post_init__", "from_json"},
        "bidding": {"makeAdvancedBid", "makeBid", "getPlayerBids",
                    "estimate_card_win_probabilities", "mini_monte_sim"},
        "gameplay": {"playFullGame", "playFullRound", "playFullTrick",
                     "makeMove", "randomDeal", "processScore"},
    }
    for label, fn_names in interesting.items():
        summarize(label, lambda f, wanted=fn_names: f[2] in wanted)
    print("TOP_CUMULATIVE")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(limit)
    print(stream.getvalue())
    print("TOP_SELF")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(limit)
    print(stream.getvalue())
    return scores, elapsed, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", choices=BOT_CLASSES, default="MonteCarloPlayer")
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--seat", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--limit", type=int, default=35)
    parser.add_argument("--output", type=Path, help="Optional cProfile .prof file")
    args = parser.parse_args()
    play_profiled_game(
        bot_name=args.bot, rollouts=args.rollouts, seed=args.seed,
        seat=args.seat, limit=args.limit, output=args.output
    )


if __name__ == "__main__":
    main()
