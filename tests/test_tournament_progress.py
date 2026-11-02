"""Live tournament progress: full-game wins and completed matched pairs."""
import pytest

from benchmarks.neural_rollout_tournament import Result, progress_report


def result(game, pair, challenger_seat, baseline_seat, scores):
    c, b, random = scores
    return Result(
        game=game, pair=pair, deal_seed=8010 + pair,
        challenger_seat=challenger_seat, baseline_seat=baseline_seat,
        random_seat=2, challenger_score=c, baseline_score=b,
        random_score=random, margin=c-b, seconds=0.25,
    )


def sample():
    return [
        result(1, 0, 0, 1, (105, 100, 50)),  # Challenger outright
        result(2, 0, 1, 0, (100, 110, 60)),  # Baseline outright
        result(3, 1, 0, 1, (100, 100, 100)), # Three-way tied first
        result(4, 1, 1, 0, (80, 70, 100)),   # Random outright
    ]


def test_progress_prints_correct_real_win_rates_and_best_rival_margin():
    line = progress_report(sample(), total_games=8, elapsed_seconds=8.0)
    assert line.startswith("PROGRESS 4/8 ")
    assert "h2h=2-1-1 " in line
    assert "h2h_win=62.5% " in line  # splits equal scores equally
    assert "first_share=33.3%/33.3% " in line  # 1 win + 1/3 tied win
    assert "outright=1-1-1 " in line
    assert "paired=1-1-0 " in line
    assert "scores=96.25/95.00 " in line
    assert "margin=+1.25 " in line
    assert "vs_best=-6.25 " in line
    assert "elapsed=8.0s eta~8s" in line
    assert "interim_ci95" not in line


def test_partial_pair_does_not_count_as_complete_pair():
    line = progress_report(sample()[:3], total_games=8, elapsed_seconds=6.0)
    assert "PROGRESS 3/8" in line
    assert "h2h=1-1-1" in line
    assert "h2h_win=50.0%" in line
    assert "paired=0-1-0" in line  # only the first complete pair


def test_twenty_game_progress_includes_provisional_paired_interval():
    # All ten pairs have identical positive mean margin, so the bootstrap CI
    # is exactly [2, 2] independently of the bootstrap RNG.
    samples = []
    for pair in range(10):
        samples.extend([
            result(2*pair + 1, pair, 0, 1, (105, 103, 70)),
            result(2*pair + 2, pair, 1, 0, (105, 103, 70)),
        ])
    line = progress_report(samples, total_games=100, elapsed_seconds=30.0,
                           seed=716, bootstrap=100)
    assert "PROGRESS 20/100" in line
    assert "paired=10-0-0" in line
    assert "interim_ci95=[+2.00,+2.00]" in line


def test_progress_rejects_invalid_pair_and_totals():
    samples = sample()
    with pytest.raises(ValueError, match="Inconsistent"):
        progress_report([samples[0], samples[2]], total_games=2,
                        elapsed_seconds=4.0)
    with pytest.raises(ValueError, match="Invalid"):
        progress_report(samples, total_games=3, elapsed_seconds=4.0)
