"""Label genuine Whist decision traces with per-action Monte Carlo histograms.

Teacher labels are distributions over 0..7 ADDITIONAL tricks won, counted
from rollouts starting at the *acting observer's* information state. They
do not use the privileged real hands from the source match.

Writes compressed, sharded .npz training tensors:
 features [N,FEATURE_DIM] float32, legal_mask [N,52] bool,
 trick_counts [N,52,8] uint32, n_rollouts [N] int32,
 plus bids/won/remaining/actual outcome/action metadata.

Split assignment is stable at the full GAME level (not by decision).
No PyTorch dependency. One complete shard is atomically renamed after save.
"""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import time
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from alphawhist.core.FastRollout import CARD_ID, evaluate_actions
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.encoding import (
    FEATURE_DIM, FEATURE_SCHEMA_VERSION, encode_decision,
)
from alphawhist.learning.recording import TRACE_SCHEMA_VERSION, iter_trace_decisions
from alphawhist.learning.timing import TimingCounters

SPLITS = ("train", "val", "test")


def _seed_for_decision(seed, rec):
    msg = json.dumps(
        [seed, rec["game_id"], rec["round"], rec["decision_in_round"],
         rec["player"]], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "big")


def _split_for_game(seed, rec):
    # Seat-swapped tournament replays share a deal_group and must remain
    # in the SAME split, preventing matched-deal train/val leakage.
    payload = f'{seed}:{rec.get("deal_group", rec["game_id"])}'.encode("utf-8")
    fraction = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    ) / 2 ** 64
    return "train" if fraction < .8 else "val" if fraction < .9 else "test"


def _label_one(rec, rollouts, workers, seed, policy, batch_size, timed=False):
    last = time.perf_counter_ns() if timed else None
    stages = {} if timed else None

    def mark(label):
        nonlocal last
        if stages is not None:
            now = time.perf_counter_ns()
            stages[label] = now - last
            last = now

    snapshot = InformationState.from_dict(rec["observation"])
    mark("deserialize_observation")
    encoded = encode_decision(snapshot)
    mark("encode_features")
    actions = tuple(rec["legal_cards"])
    if set(actions) != set(snapshot.own_hand).intersection(actions):
        raise ValueError("Training actions contradict own hand")
    if set(CARD_ID[c] for c in actions) != set(np.flatnonzero(encoded.legal_mask)):
        raise ValueError("Recorded legal actions contradict encoded legal mask")

    rollout_seed = _seed_for_decision(seed, rec)
    evaluation = evaluate_actions(
        snapshot, actions, n_rollouts=rollouts, seed=rollout_seed,
        workers=workers, batch_size=batch_size, policy=policy,
        collect_trick_histograms=True,
    )
    mark("monte_carlo")
    labels = np.zeros((52, 8), dtype=np.uint32)
    for action, counts, expected_score in zip(
            evaluation.actions, evaluation.additional_trick_counts,
            evaluation.values):
        if sum(counts) != rollouts:
            raise AssertionError("Incomplete Monte Carlo teacher label")
        labels[CARD_ID[action]] = counts
        # Regression: the categorical label must reconstruct the identical
        # expected-score estimate returned by the existing MC evaluator.
        derived = sum(
            n * (snapshot.tricks_won[snapshot.observer_id] + k
                 + 10 * (snapshot.tricks_won[snapshot.observer_id] + k
                         == snapshot.bids[snapshot.observer_id]))
            for k, n in enumerate(counts)
        ) / rollouts
        if abs(derived - expected_score) > 1e-8:
            raise AssertionError("Trick histogram and score teacher disagree")
    mark("validate_and_pack_labels")
    result = {
        "features": encoded.features,
        "legal_mask": encoded.legal_mask,
        "trick_counts": labels,
        "n_rollouts": rollouts,
        "remaining_tricks": encoded.remaining_tricks,
        "tricks_won": encoded.tricks_won,
        "bids": encoded.bid,
        "chosen_card_id": CARD_ID[rec["chosen_card"]],
        "actual_remaining_tricks": rec["actual_remaining_tricks_won"],
        "actual_round_score": rec["actual_round_score"],
        "game_id": rec["game_id"],
        "round": rec["round"],
        "decision_in_round": rec["decision_in_round"],
        "player": rec["player"],
        "teacher_seed": rollout_seed,
    }
    if stages is not None:
        result["_timing_ns"] = stages
    return result


def _label_state_batch(task):
    """Run a small block of states inside one persistent worker process.

    Rollout workers MUST be one: parallelism happens across independent
    decision states, never with nested per-state process pools.
    """
    records, rollouts, seed, policy, batch_size, *options = task
    timed = bool(options[0]) if options else False
    return [
        (split, _label_one(rec, rollouts, 1, seed, policy, batch_size, timed))
        for split, rec in records
    ]


def _record_batches(records, state_batch_size):
    """Yield bounded blocks from a streaming JSONL decision iterator."""
    iterator = iter(records)
    while batch := tuple(islice(iterator, state_batch_size)):
        yield batch


def _iter_labeled_states(records, rollouts, workers, seed, policy,
                         batch_size, state_batch_size, prefetch_batches,
                         timed=False):
    """Stream deterministic labels, preserving input order across workers.

    At most workers * prefetch_batches * state_batch_size decisions are
    submitted at once. This avoids ProcessPoolExecutor.map() on Python 3.11,
    which can eagerly consume the entire source iterator. A single pool
    survives for all states; workers always run evaluate_actions(workers=1).
    """
    if workers == 1:
        for split, rec in records:
            yield split, _label_one(rec, rollouts, 1, seed, policy, batch_size,
                                    timed)
        return

    batches = iter(_record_batches(records, state_batch_size))
    max_pending = workers * prefetch_batches
    pending = deque()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        def submit_next():
            try:
                batch = next(batches)
            except StopIteration:
                return False
            pending.append(pool.submit(
                _label_state_batch,
                (batch, rollouts, seed, policy, batch_size, timed),
            ))
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            # Submission order ensures bitwise-identical row/shard ordering
            # regardless of worker count or individual state difficulty.
            for item in pending.popleft().result():
                yield item
            submit_next()


def _write_shard(directory, split, index, rows):
    if not rows:
        return None
    path = directory / f"{split}-{index:05d}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite training shard: {path}")
    temporary = path.with_name(path.name + ".part")
    arrays = {
        "features": np.stack([r["features"] for r in rows]),
        "legal_mask": np.stack([r["legal_mask"] for r in rows]),
        "trick_counts": np.stack([r["trick_counts"] for r in rows]),
    }
    for key in ("n_rollouts", "remaining_tricks", "tricks_won", "bids",
                "chosen_card_id", "actual_remaining_tricks",
                "actual_round_score", "round", "decision_in_round", "player"):
        arrays[key] = np.asarray([r[key] for r in rows], dtype=np.int32)
    arrays["game_id"] = np.asarray([r["game_id"] for r in rows], dtype="U96")
    arrays["teacher_seed"] = np.asarray(
        [r["teacher_seed"] for r in rows], dtype=np.uint64
    )
    # Use a file handle to prevent NumPy from appending .npz to .part.
    try:
        with temporary.open("wb") as target:
            np.savez_compressed(target, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def label_traces(paths, output_dir, rollouts=1024, workers=1,
                 seed=20261102, policy="bid_aware_2", batch_size=256,
                 shard_size=256, max_states=None, include_forced=False,
                 state_batch_size=4, prefetch_batches=2,
                 timings_json=None):
    if (rollouts < 1 or workers < 1 or batch_size < 1 or shard_size < 1
            or state_batch_size < 1 or prefetch_batches < 1):
        raise ValueError("Rollout, worker, shard and prefetch counts must be positive")
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive")
    if policy not in ("random", "highest", "bid_aware_2"):
        raise ValueError("Invalid teacher continuation policy")
    paths = sorted(map(Path, paths))
    if not paths:
        raise ValueError("No input traces")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Output directory must be empty (never overwrite labels)")

    started = time.perf_counter()
    timings = TimingCounters() if timings_json is not None else None
    buffers = defaultdict(list)
    counts = dict.fromkeys(SPLITS, 0)
    indices = dict.fromkeys(SPLITS, 0)
    processed = 0
    shard_names = []
    manifest_path = out / "manifest.json"

    def flush(split):
        if not buffers[split]:
            return
        before_write = time.perf_counter_ns() if timings is not None else None
        path = _write_shard(out, split, indices[split], buffers[split])
        if timings is not None:
            timings.add("parent.shard_compression_and_write",
                        time.perf_counter_ns() - before_write)
        indices[split] += 1
        shard_names.append(path.name)
        buffers[split].clear()

    source = iter_trace_decisions(paths, include_forced=include_forced)
    if max_states is not None:
        source = islice(source, max_states)

    # Attach the assigned split before handing records to the workers.
    # Each worker returns its (split, label) pairs in exactly source order.
    for split, row in _iter_labeled_states(
            ((_split_for_game(seed, rec), rec) for rec in source),
            rollouts, workers, seed, policy, batch_size,
            state_batch_size, prefetch_batches,
            timings is not None):
        if timings is not None:
            for stage, nanoseconds in row.pop("_timing_ns").items():
                timings.add("worker." + stage, nanoseconds)
        buffers[split].append(row)
        counts[split] += 1
        processed += 1
        if len(buffers[split]) >= shard_size:
            flush(split)
            print(f"SHARD {split} index={indices[split] - 1} "
                  f"states={processed}", flush=True)
    for split in SPLITS:
        flush(split)
    manifest = {
        "dataset_schema": 1,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "trace_schema": TRACE_SCHEMA_VERSION,
        "input_traces": [str(p) for p in paths],
        "teacher_policy": policy,
        "rollouts_per_action": rollouts, "teacher_seed": seed,
        "shard_size": shard_size, "counts_by_split": counts,
        "shards": shard_names, "include_forced": include_forced,
        "total_states": processed,
        "workers": workers,
        "parallelism": "persistent state-level pool, one MC worker per state",
        "state_batch_size": state_batch_size,
        "prefetch_batches": prefetch_batches,
        "split_assignment": "80/10/10 deterministic per deal_group",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    if timings is not None:
        timings.save(timings_json, {
            "states": processed, "workers": workers, "rollouts": rollouts,
            "teacher_policy": policy, "wall_seconds": elapsed,
            "note": "Worker durations summed across processes are NOT "
                    "end-to-end wall time. Timed mode has additional overhead.",
        })
    print(f"DONE labeled_states={processed} splits={counts} "
          f"workers={workers} state_batch_size={state_batch_size} "
          f"seconds={elapsed:.3f} states_per_second={processed / elapsed:.3f} "
          f"manifest={manifest_path}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True,
                        help="JSONL trace paths or quoted glob patterns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--policy", default="bid_aware_2",
                        choices=("random", "highest", "bid_aware_2"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--include-forced", action="store_true")
    parser.add_argument("--state-batch-size", type=int, default=4,
                        help="States per worker task, amortizing IPC")
    parser.add_argument("--prefetch-batches", type=int, default=2,
                        help="Outstanding tasks per worker (bounded streaming)")
    parser.add_argument("--timings-json", type=Path,
                        help="Opt-in per-stage timing summary, no per-row timing storage")
    args = parser.parse_args()
    traces = sorted(set(p for pattern in args.traces for p in glob.glob(pattern)))
    label_traces(traces, args.output_dir, args.rollouts, args.workers,
                 args.seed, args.policy, args.batch_size, args.shard_size,
                 args.max_states, args.include_forced,
                 args.state_batch_size, args.prefetch_batches,
                 args.timings_json)


if __name__ == "__main__":
    main()
