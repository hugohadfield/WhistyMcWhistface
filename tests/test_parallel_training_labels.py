"""Regression: persistent state-level parallel labeling is bit-for-bit reproducible.

Do not assert wall-time speedup in CI: runners have variable and often limited
CPU quotas. The exporter reports actual states/second for field benchmarks.
"""
import contextlib
import io
import json
from itertools import islice

import numpy as np
import pytest

from alphawhist.learning.recording import iter_trace_decisions
from benchmarks.generate_training_games import generate
from benchmarks.export_training_labels import (
    _iter_labeled_states, _label_state_batch, _record_batches, label_traces,
)


def _trace_fixture(tmp_path):
    """A compact, multi-deal fixture; split groups remain independent."""
    raw = tmp_path / "raw"
    with contextlib.redirect_stdout(io.StringIO()):
        generate(games=4, seed=20265411, players=3, agents=("random",),
                 workers=2, output_dir=raw)
    records = []
    for file in sorted(raw.glob("*.jsonl")):
        records.extend(islice(iter_trace_decisions([file]), 4))
    path = tmp_path / "decisions.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for item in records:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    assert len(records) == 16
    assert len(set(r["deal_group"] for r in records)) == 4
    return path, records


def _shards(out):
    contents = {}
    for file in sorted(out.glob("*.npz")):
        with np.load(file) as arrays:
            contents[file.name] = {
                name: np.array(arrays[name]) for name in arrays.files
            }
    return contents


@pytest.mark.parametrize("workers,state_batch_size,prefetch", [
    (2, 1, 1), (2, 3, 2), (3, 4, 3), (4, 7, 1),
])
def test_parallel_export_identical_to_serial_for_multiple_deals(
        tmp_path, workers, state_batch_size, prefetch):
    trace, _ = _trace_fixture(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    common = dict(rollouts=19, seed=481, policy="bid_aware_2",
                  batch_size=7, shard_size=3, max_states=15)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = label_traces([trace], serial, workers=1, **common)
        result = label_traces([trace], parallel, workers=workers,
                              state_batch_size=state_batch_size,
                              prefetch_batches=prefetch, **common)
    assert ref["total_states"] == result["total_states"] == 15
    assert ref["counts_by_split"] == result["counts_by_split"]
    assert ref["shards"] == result["shards"]
    left, right = _shards(serial), _shards(parallel)
    assert left.keys() == right.keys()
    for shard_name in left:
        assert left[shard_name].keys() == right[shard_name].keys()
        for name in left[shard_name]:
            np.testing.assert_array_equal(left[shard_name][name],
                                          right[shard_name][name])


@pytest.mark.parametrize("batch_size", [1, 2, 5, 17])
def test_streaming_batching_preserves_input_order_and_tail(batch_size):
    records = list(range(13))
    actual = [item for batch in _record_batches(records, batch_size)
              for item in batch]
    assert actual == records
    assert list(_record_batches([], batch_size)) == []


def test_label_batch_matches_serial_and_has_no_nested_pools(tmp_path):
    trace, records = _trace_fixture(tmp_path)
    items = [("train", r) for r in records[:5]]
    grouped = _label_state_batch((items, 11, 42, "random", 3))
    serial = list(_iter_labeled_states(
        items, 11, 1, 42, "random", 3, 2, 2,
    ))
    assert len(grouped) == len(serial)
    for a, b in zip(grouped, serial):
        assert a[0] == b[0]
        for key, value in a[1].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, b[1][key])
            else:
                assert value == b[1][key]


@pytest.mark.parametrize("workers,batch,prefetch", [
    (0, 2, 2), (2, 0, 2), (2, 2, 0),
])
def test_reject_bad_parallel_settings(tmp_path, workers, batch, prefetch):
    with pytest.raises(ValueError, match="positive"):
        label_traces([tmp_path / "unused.jsonl"], tmp_path / "out",
                     workers=workers, state_batch_size=batch,
                     prefetch_batches=prefetch)


def test_max_states_is_exact_even_with_prefetch(tmp_path):
    trace, _ = _trace_fixture(tmp_path)
    out = tmp_path / "limited"
    with contextlib.redirect_stdout(io.StringIO()):
        result = label_traces(
            [trace], out, workers=4, rollouts=7, seed=5,
            state_batch_size=7, prefetch_batches=3,
            max_states=3, shard_size=2,
        )
    assert result["total_states"] == 3
    assert sum(result["counts_by_split"].values()) == 3
    assert sum(len(v["features"]) for v in _shards(out).values()) == 3
