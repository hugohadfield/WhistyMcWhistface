"""Reproducible card-evaluator microbenchmark smoke tests."""
import pytest
from benchmarks.measure_evaluator import measure


@pytest.mark.parametrize("scenario", ["early", "mid"])
def test_seeded_evaluator_benchmark_stable(scenario, capsys):
    durations, result = measure(scenario=scenario, rollouts=8, repeats=2, seed=121)
    assert len(durations) == 2
    assert all(x >= 0 for x in durations)
    assert result.best_action() in result.actions
    out = capsys.readouterr().out
    assert "EVALUATOR scenario=" + scenario in out
    assert "scores=" in out


def test_evaluator_benchmark_rejects_invalid_counts():
    with pytest.raises(ValueError):
        measure(rollouts=0)
    with pytest.raises(ValueError):
        measure(repeats=0)
