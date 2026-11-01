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
from collections import defaultdict
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

SPLITS = ("train", "val", "test")


def _seed_for_decision(seed, rec):
    msg = json.dumps(
        [seed, rec["game_id"], rec["round"], rec["decision_in_round"],
         rec["player"]], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "big")


def _split_for_game(seed, rec):
    # Group by run seed/index in game_id, not by the position within a game.
    payload = f'{seed}:{rec["game_id"]}'.encode("utf-8")
    fraction = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    ) / 2 ** 64
    return "train" if fraction < .8 else "val" if fraction < .9 else "test"


def _label_one(rec, rollouts, workers, seed, policy, batch_size):
    snapshot = InformationState.from_dict(rec["observation"])
    encoded = encode_decision(snapshot)
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
    return {
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
                 shard_size=256, max_states=None, include_forced=False):
    if rollouts < 1 or workers < 1 or batch_size < 1 or shard_size < 1:
        raise ValueError("Invalid labelling budget or shard size")
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

    buffers = defaultdict(list)
    counts = dict.fromkeys(SPLITS, 0)
    indices = dict.fromkeys(SPLITS, 0)
    processed = 0
    shard_names = []
    manifest_path = out / "manifest.json"

    def flush(split):
        if not buffers[split]:
            return
        path = _write_shard(out, split, indices[split], buffers[split])
        indices[split] += 1
        shard_names.append(path.name)
        buffers[split].clear()

    for rec in iter_trace_decisions(paths, include_forced=include_forced):
        split = _split_for_game(seed, rec)
        buffers[split].append(_label_one(
            rec, rollouts, workers, seed, policy, batch_size,
        ))
        counts[split] += 1
        processed += 1
        if len(buffers[split]) >= shard_size:
            flush(split)
            print(f"SHARD {split} index={indices[split] - 1} "
                  f"states={processed}", flush=True)
        if max_states is not None and processed >= max_states:
            break
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
        "split_assignment": "80/10/10 deterministic per game_id",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DONE labeled_states={processed} splits={counts} "
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
    args = parser.parse_args()
    traces = sorted(set(p for pattern in args.traces for p in glob.glob(pattern)))
    label_traces(traces, args.output_dir, args.rollouts, args.workers,
                 args.seed, args.policy, args.batch_size, args.shard_size,
                 args.max_states, args.include_forced)


if __name__ == "__main__":
    main()
