"""Smoke-test the reproducible full-match win-rate experiment."""
import pytest

from benchmarks.ten_games_vs_random import (
    BOTS, MatchResult, play_match, run_experiment,
)


@pytest.mark.parametrize("bot", list(BOTS.values()))
def test_full_game_match_outcome_matches_final_scores(bot):
    match = play_match(bot, seed=925, seat=1, rollouts=4)
    assert isinstance(match, MatchResult)
    assert len(match.scores) == 3
    assert match.seconds >= 0
    assert match.ai_score == match.scores[1]
    is_sole_winner = match.ai_score > max(match.rival_scores)
    is_tied_lead = match.ai_score == max(match.rival_scores)
    assert match.outcome == (
        "WIN" if is_sole_winner else "TIE" if is_tied_lead else "LOSS"
    )


def test_match_runner_prints_summary_and_rotates_seats(capsys):
    results = run_experiment(
        games=3, rollouts=4, seed=933, bots=("MonteCarloPlayer",)
    )
    matches = results["MonteCarloPlayer"]
    assert [m.seat for m in matches] == [0, 1, 2]
    assert [m.seed for m in matches] == [933, 934, 935]
    assert all(len(m.scores) == 3 for m in matches)
    out = capsys.readouterr().out
    assert "SUMMARY bot=MonteCarloPlayer" in out
    assert "games=3" in out


def test_match_runner_rejects_bad_settings():
    with pytest.raises(ValueError):
        run_experiment(games=0)
    with pytest.raises(ValueError):
        play_match(BOTS["MonteCarloPlayer"], seed=1, seat=7, rollouts=4)
    with pytest.raises(ValueError):
        play_match(BOTS["MonteCarloPlayer"], seed=1, seat=0, rollouts=0)
