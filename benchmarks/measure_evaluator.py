"""Measure the same seeded card-decision workload on multiple revisions.

When run on the baseline and optimized checkout with identical flags,
the reported score vectors and best action must be *identical*; compare
median wall times to isolate simulation-engine changes from bidding/RNG
effects elsewhere in a full game.

Examples:
    python -m benchmarks.measure_evaluator --scenario early --rollouts 4096 --repeats 5
    python -m benchmarks.measure_evaluator --scenario mid --rollouts 4096 --repeats 5
"""
import argparse
import statistics
import time

from benchmarks.score_convergence import example_snapshot
from alphawhist.core.FastRollout import evaluate_actions


def measure(scenario="early", rollouts=4096, repeats=5, seed=12345):
    if rollouts < 1 or repeats < 1:
        raise ValueError("Rollouts and repeats must be positive")
    state = example_snapshot(scenario)
    durations = []
    last = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = evaluate_actions(
            state, n_rollouts=rollouts, seed=seed, workers=1
        )
        durations.append(time.perf_counter() - started)
        if last is not None:
            assert result == last, "Repeated seeded evaluations disagree"
        last = result
    assert last is not None
    scores = [tuple(round(x, 8) for x in row)
              for row in last.mean_round_scores]
    print(
        f"EVALUATOR scenario={scenario} rollouts={rollouts} "
        f"actions={len(last.actions)} repeats={repeats} "
        f"median_sec={statistics.median(durations):.6f} "
        f"min_sec={min(durations):.6f} "
        f"best={last.best_action()} "
        f"cards={last.actions} scores={scores}",
        flush=True,
    )
    return durations, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("early", "mid"), default="early")
    parser.add_argument("--rollouts", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    measure(args.scenario, args.rollouts, args.repeats, args.seed)


if __name__ == "__main__":
    main()
