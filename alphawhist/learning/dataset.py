"""Streaming, validated sharded data for distributional Whist training.

Never reads privileged replay hands. Load one .npz shard at a time, with
deterministic within-shard shuffle. Splits must be provided by exporter
manifest; never randomly split correlated decisions.
"""
import json
from pathlib import Path

import numpy as np

from alphawhist.learning.encoding import FEATURE_DIM, FEATURE_SCHEMA_VERSION

REQUIRED = {
    "features", "legal_mask", "trick_counts", "n_rollouts",
    "remaining_tricks", "tricks_won", "bids", "game_id",
}


def read_manifest(dataset):
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_schema") != 1:
        raise ValueError("Unsupported dataset schema")
    if manifest.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema mismatch; cannot train old representation")
    if not isinstance(manifest.get("teacher_policy"), str):
        raise ValueError("Missing teacher continuation policy")
    files = manifest.get("shards")
    if not isinstance(files, list) or len(files) != len(set(files)):
        raise ValueError("Invalid or duplicate shards in manifest")
    if set(manifest.get("counts_by_split", {})) != {"train", "val", "test"}:
        raise ValueError("Expected explicit train, val and test counts")
    for name in files:
        if (Path(name).name != name or not name.endswith(".npz")
                or not name.startswith(("train-", "val-", "test-"))
                or not (root / name).is_file()):
            raise ValueError(f"Missing or unsafe shard {name}")
    if sum(manifest["counts_by_split"].values()) != manifest.get("total_states"):
        raise ValueError("Incorrect manifest split counts")
    return manifest


def split_files(root, manifest, split, require_nonempty=True):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    files = sorted(Path(root) / x for x in manifest["shards"]
                   if x.startswith(split + "-"))
    if require_nonempty and (not files or manifest["counts_by_split"][split] == 0):
        raise ValueError(f"Empty {split} split: generate more independent games")
    return files


def load_shard(path):
    """Read and validate a single compressed shard, rejecting corrupt labels."""
    with np.load(path, allow_pickle=False) as data:
        if not REQUIRED.issubset(data.files):
            raise ValueError(f"Missing arrays in {path}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED}
    f, legal, counts = (arrays[k] for k in
                        ("features", "legal_mask", "trick_counts"))
    n = len(f)
    if (n < 1 or f.shape != (n, FEATURE_DIM)
            or legal.shape != (n, 52) or counts.shape != (n, 52, 8)):
        raise ValueError("Invalid neural tensor shapes")
    if legal.dtype != np.bool_ or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Wrong mask or count dtype")
    if not np.issubdtype(f.dtype, np.floating) or not np.isfinite(f).all():
        raise ValueError("Features must be finite floats")
    for field in ("n_rollouts", "remaining_tricks", "tricks_won", "bids", "game_id"):
        if arrays[field].shape != (n,):
            raise ValueError(f"Invalid {field} shape")
    if not legal.any(axis=1).all() or (counts < 0).any():
        raise ValueError("Bad legal action mask or counts")
    budget = arrays["n_rollouts"].astype(np.int64)
    remaining = arrays["remaining_tricks"].astype(np.int64)
    if (budget <= 0).any() or (remaining < 1).any() or (remaining > 7).any():
        raise ValueError("Invalid rollout or remaining-trick count")
    if not np.array_equal(counts.sum(axis=-1), legal * budget[:, None]):
        raise ValueError("MC histograms must sum to rollout budgets for legal cards")
    if (counts * (np.arange(8)[None, None, :] >
                  remaining[:, None, None])).any():
        raise ValueError("Impossible additional-trick label")
    return arrays


def shard_batches(files, batch_size, seed=None):
    """Iterate one shard's arrays at a time; shuffle deterministically in train."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    files = list(files)
    rng = np.random.default_rng(seed)
    if seed is not None:
        rng.shuffle(files)
    for file in files:
        data = load_shard(file)
        order = np.arange(len(data["features"]))
        if seed is not None:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            yield {name: values[ids] for name, values in data.items()}


def to_tensors(batch, device):
    import torch

    result = {}
    for key in ("features", "legal_mask", "trick_counts",
                "n_rollouts", "remaining_tricks", "tricks_won", "bids"):
        arr = batch[key]
        dtype = (torch.bool if key == "legal_mask" else
                 torch.float32 if key in ("features", "trick_counts") else torch.long)
        result[key] = torch.as_tensor(arr, device=device, dtype=dtype)
    result["targets"] = (
        result["trick_counts"] / result["n_rollouts"][:, None, None]
    )
    return result
