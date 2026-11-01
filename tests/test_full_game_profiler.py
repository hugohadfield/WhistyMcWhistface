"""Smoke tests for full-match profiler output and saved cProfile file."""
import contextlib
import io
import pstats

import pytest

from benchmarks.profile_full_game import play_profiled_game


@pytest.mark.parametrize("bot", ["MonteCarloPlayer", "SnakeMonteCarloPlayer"])
def test_full_game_profiler_produces_valid_profile(bot, tmp_path):
    path = tmp_path / "match.prof"
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        scores, elapsed, stats = play_profiled_game(
            bot_name=bot, rollouts=2, seed=20261031,
            seat=0, limit=5, output=path
        )
    assert len(scores) == 3
    assert all(score >= 0 for score in scores)
    assert elapsed >= 0
    assert path.exists() and path.stat().st_size > 0
    pstats.Stats(str(path))
    out = stdout.getvalue()
    assert "PROFILE_MATCH bot=" in out
    assert "KEY_HOTSPOTS" in out
    assert "TOP_CUMULATIVE" in out
    assert "TOP_SELF" in out


@pytest.mark.parametrize("kwargs", [
    {"rollouts": 0}, {"seat": 9}, {"limit": 0}, {"bot_name": "unknown"}
])
def test_profiler_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        play_profiled_game(**kwargs)
