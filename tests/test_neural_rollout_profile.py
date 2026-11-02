"""The profiler measures real (not synthetic) neural-guided game decisions."""
import json
import pstats

import pytest

torch = pytest.importorskip("torch")

from alphawhist.learning.checkpoint import save_checkpoint
from alphawhist.learning.model import DistributionalWhistMLP
from benchmarks.profile_neural_rollout import grouped_hotspots, profile_game


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(399)
    network = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, network, metadata={"teacher_policy": "bid_aware_2"})
    return path


def test_one_real_round_writes_cpu_profile_and_hotspot_report(checkpoint, tmp_path):
    prefix = tmp_path / "profile"
    report = profile_game(checkpoint, rounds=1, rollouts=2,
                          worlds_per_batch=2, output_prefix=prefix,
                          seed=90061101)
    assert report["rounds"] == 1
    assert report["rollouts_per_candidate"] == 2
    assert report["challenger_decisions"] == 7
    assert report["challenger_nonforced_searches"] > 0
    assert report["search_wall_seconds_under_profiler"] > 0
    assert report["game_wall_seconds_under_profiler"] >= (
        report["search_wall_seconds_under_profiler"])
    assert (tmp_path / "profile.prof").is_file()
    assert (tmp_path / "profile.txt").is_file()
    assert (tmp_path / "profile.json").is_file()
    restored = json.loads((tmp_path / "profile.json").read_text())
    assert restored["hotspots"] == report["hotspots"]
    assert "CUMULATIVE TIME" in (tmp_path / "profile.txt").read_text()
    # The raw binary profile is standard pstats-compatible.
    assert pstats.Stats(str(tmp_path / "profile.prof")).total_calls > 0
    stages = {item["stage"]: item for item in report["hotspots"]}
    for name in ("advance public observation", "public event reducer",
                 "encode one observation", "neural inference",
                 "simulate one legal card"):
        assert stages[name]["calls"] > 0


def test_bad_profile_configuration_rejected(checkpoint, tmp_path):
    for kwargs in ({"rollouts": 0}, {"rounds": 0}, {"rounds": 14},
                   {"worlds_per_batch": 0}):
        with pytest.raises(ValueError):
            profile_game(checkpoint, output_prefix=tmp_path / "bad", **kwargs)
    with pytest.raises(FileNotFoundError):
        profile_game(tmp_path / "missing.pt")
