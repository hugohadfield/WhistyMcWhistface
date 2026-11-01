"""Detailed CPU profile of Whist game trace collection and MC teacher labels.

Runs a separate cProfile single-process trace for the two hot pipelines,
uses a stratified sample spanning 2..7 remaining cards (non-forced actions), and benchmarks the
identical teacher workload with 1/2/4 persistent state-parallel workers.
Produces .prof binary pstats, human-readable top-function tables, stage
timing JSON and a summary JSON. No GPUs or private hands required.

cProfile heavily perturbs timings; use the separately timed, unprofiled
runs for throughput comparison. Never interpret summed worker seconds as
elapsed wall time.
"""
import argparse
import contextlib
import cProfile
import io
import json
from collections import defaultdict
from pathlib import Path
import pstats
import time

from alphawhist.learning.recording import iter_trace_decisions
from benchmarks.export_training_labels import label_traces
from benchmarks.generate_training_games import play_training_game


def _cprofile(name, directory, fn, *args, **kwargs):
    statsfile = directory / f"{name}.prof"
    profile = cProfile.Profile()
    before = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        profile.enable()
        result = fn(*args, **kwargs)
        profile.disable()
    elapsed = time.perf_counter() - before
    profile.dump_stats(str(statsfile))
    tables = []
    for criterion in ("cumulative", "tottime"):
        out = io.StringIO()
        pstats.Stats(profile, stream=out).strip_dirs().sort_stats(criterion).print_stats(35)
        tables.append("SORT " + criterion + "\n" + out.getvalue())
    (directory / f"{name}-top-functions.txt").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )
    return result, elapsed


def _select_diverse_traces(paths, output, states_per_size):
    by_size = defaultdict(list)
    for rec in iter_trace_decisions(paths):
        snap = rec["observation"]
        size = snap["cards_per_player"] - snap["trick_number"]
        if size in range(2, 8) and len(by_size[size]) < states_per_size:
            by_size[size].append(rec)
    for size in range(2, 8):
        if not by_size[size]:
            raise ValueError(f"No states found with {size} cards remaining")
    with output.open("w", encoding="utf-8") as stream:
        for size in sorted(by_size, reverse=True):
            for item in by_size[size]:
                stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    return {str(size): len(items) for size, items in sorted(by_size.items())}


def run_profile(output_dir, behavior_rollouts=16, teacher_rollouts=256,
                states_per_size=6, workers=(1, 2, 4), seed=20261101):
    if min(behavior_rollouts, teacher_rollouts, states_per_size) < 1:
        raise ValueError("Rollout budgets and number of states must be positive")
    if not workers or any(n < 1 for n in workers) or len(set(workers)) != len(workers):
        raise ValueError("Worker counts must be positive and distinct")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Profile output directory must be empty")

    traces = out / "traces"
    traces.mkdir()
    game_timing = out / "game-timing"
    game_timing.mkdir()
    collection = {}
    for index, agents in [
        (0, ("random",)),
        (1, ("random", "mc_random", "mc_aware")),
    ]:
        name = "random-game" if index == 0 else "mixed-game"
        result, elapsed = _cprofile(
            name, out, play_training_game,
            index, seed, 3, agents, behavior_rollouts, traces,
            None, game_timing,
        )
        collection[name] = {
            "seconds_with_cprofile": elapsed,
            "decisions": result[1],
            "agent_mix": list(agents),
            "timings_file": f"game-timing/game-{index:07d}.json",
        }
        print(f"PROFILE {name} seconds={elapsed:.3f} decisions={result[1]}", flush=True)

    selected_path = out / "stratified-decisions.jsonl"
    sizes = _select_diverse_traces(
        sorted(traces.glob("game-*.jsonl")), selected_path, states_per_size,
    )
    total = sum(sizes.values())
    if total < 6:
        raise AssertionError("Stratified sample is too small")
    common = dict(
        rollouts=teacher_rollouts, policy="bid_aware_2",
        seed=seed ^ 0x11771, shard_size=16, batch_size=128,
        state_batch_size=4, prefetch_batches=2,
    )
    # Worker profiling happens in the same process to expose the real
    # Python-level sampling/rollout hot functions in cProfile.
    serial_profiled = out / "labels-cprofile"
    _, profiled_seconds = _cprofile(
        "teacher-cprofile", out, label_traces,
        [selected_path], serial_profiled, workers=1,
        timings_json=out / "teacher-cprofile-stage-timing.json",
        **common,
    )
    results = {}
    for worker_count in workers:
        labeled_dir = out / f"labels-workers-{worker_count}"
        before = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = label_traces(
                [selected_path], labeled_dir,
                workers=worker_count,
                timings_json=out / f"teacher-timing-workers-{worker_count}.json",
                **common,
            )
        seconds = time.perf_counter() - before
        results[str(worker_count)] = {
            "seconds": seconds,
            "states_per_second": total / seconds,
            "manifest": str(labeled_dir / "manifest.json"),
            "state_count": manifest["total_states"],
            "timings_file": f"teacher-timing-workers-{worker_count}.json",
        }
        print(f"THROUGHPUT workers={worker_count} states={total} "
              f"seconds={seconds:.3f} states_per_second={total / seconds:.3f}",
              flush=True)

    # Validate deterministic label data as an integral part of the profile.
    import numpy as np
    reference = out / f"labels-workers-{workers[0]}"
    names = sorted(p.name for p in reference.glob("*.npz"))
    if not names:
        raise AssertionError("No labeled dataset shards")
    for worker_count in workers[1:]:
        other = out / f"labels-workers-{worker_count}"
        if names != sorted(p.name for p in other.glob("*.npz")):
            raise AssertionError("Worker counts generated different shards")
        for name in names:
            with np.load(reference / name) as a, np.load(other / name) as b:
                if a.files != b.files:
                    raise AssertionError("Different shard keys")
                for field in a.files:
                    np.testing.assert_array_equal(a[field], b[field])

    summary = {
        "profile_schema": 1,
        "behavior_rollouts": behavior_rollouts,
        "teacher_rollouts": teacher_rollouts,
        "seed": seed,
        "states_by_remaining_cards": sizes,
        "states_total": total,
        "game_collection": collection,
        "labeling_cprofile_wall_seconds": profiled_seconds,
        "unprofiled_labeling": results,
        "note": "cProfile perturbs timings; compare unprofiled worker runs",
    }
    (out / "profile-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PROFILE_DONE states={total} output={out}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-rollouts", type=int, default=16)
    parser.add_argument("--teacher-rollouts", type=int, default=256)
    parser.add_argument("--states-per-size", type=int, default=6)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=20261101)
    a = parser.parse_args()
    run_profile(a.output_dir, a.behavior_rollouts, a.teacher_rollouts,
                a.states_per_size, tuple(a.workers), a.seed)


if __name__ == "__main__":
    main()
