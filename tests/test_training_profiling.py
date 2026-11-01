"""The optional profilers must not change data, decisions or teacher labels."""
import contextlib
import io
import json

import numpy as np

from alphawhist.learning.timing import TimingCounters
from benchmarks.generate_training_games import play_training_game
from benchmarks.export_training_labels import label_traces


def test_constant_memory_timing_counters_and_merge(tmp_path):
    a, b = TimingCounters(), TimingCounters()
    for value in (10, 100, 1_000, 10_000):
        a.add("step", value)
    b.add("step", 40)
    a.merge(b)
    row = a.summary()["step"]
    assert row["count"] == 5
    assert row["total_seconds"] == 11150 / 1e9
    assert row["max_ms"] == .01
    assert row["p50_ms_upper_bound"] > 0
    a.save(tmp_path / "timing.json", {"mode": "test"})
    result = json.loads((tmp_path / "timing.json").read_text())
    assert result["metadata"]["mode"] == "test"
    assert result["timing"]["step"]["count"] == 5


def test_game_instrumentation_does_not_change_public_trace(tmp_path):
    base = tmp_path / "base"
    timed = tmp_path / "timed"
    timings = tmp_path / "timings"
    a = play_training_game(0, seed=2026891, players=3,
                           agents=("random",), rollouts=3,
                           output_dir=base)
    b = play_training_game(0, seed=2026891, players=3,
                           agents=("random",), rollouts=3,
                           output_dir=timed, timings_dir=timings)
    assert a[:2] == b[:2]
    assert (base / "game-0000000.jsonl").read_bytes() == (
        timed / "game-0000000.jsonl").read_bytes()
    meta = json.loads((timings / "game-0000000.json").read_text())
    assert meta["metadata"]["trace_bytes"] > 0
    stages = meta["timing"]
    assert stages["game.observe"]["count"] == 165
    assert stages["policy.random.makeMove"]["count"] == 165
    assert stages["recording.on_decision"]["count"] == 165
    assert stages["recording.on_round_end"]["count"] == 13
    assert stages["game.randomDeal"]["count"] == 13


def test_label_stage_timing_identical_to_untimed_shards(tmp_path):
    source = tmp_path / "trace"
    play_training_game(0, seed=2026892, players=3, agents=("random",),
                       rollouts=3, output_dir=source)
    common = dict(
        paths=list(source.glob("*.jsonl")), rollouts=11, policy="random",
        seed=2041, shard_size=3, max_states=9, batch_size=4,
    )
    a, b, c = (tmp_path / "a", tmp_path / "b", tmp_path / "c")
    with contextlib.redirect_stdout(io.StringIO()):
        x = label_traces(output_dir=a, workers=1, **common)
        y = label_traces(output_dir=b, workers=1,
                         timings_json=tmp_path / "timed1.json", **common)
        z = label_traces(output_dir=c, workers=2,
                         timings_json=tmp_path / "timed2.json", **common)
    assert x["shards"] == y["shards"] == z["shards"]
    for name in x["shards"]:
        with np.load(a/name) as one, np.load(b/name) as two, np.load(c/name) as three:
            for key in one.files:
                np.testing.assert_array_equal(one[key], two[key])
                np.testing.assert_array_equal(one[key], three[key])
    for file in (tmp_path / "timed1.json", tmp_path / "timed2.json"):
        data = json.loads(file.read_text())
        stages = data["timing"]
        for key in (
            "worker.deserialize_observation", "worker.encode_features",
            "worker.monte_carlo", "worker.validate_and_pack_labels",
            "parent.shard_compression_and_write",
        ):
            assert key in stages
            assert stages[key]["total_seconds"] >= 0
        assert stages["worker.monte_carlo"]["count"] == 9
        assert stages["parent.shard_compression_and_write"]["count"] == len(x["shards"])
