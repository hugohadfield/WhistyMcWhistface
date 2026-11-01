"""Regression for paired bid-aware-vs-random rollout tournament."""
from dataclasses import replace

import pytest

from benchmarks.endgame_policy_tournament import play_match, summary, tournament


@pytest.mark.parametrize("index", range(6))
def test_complete_13_round_match_and_seat_swap(index):
    m = play_match(index, seed=20261461, rollouts=8)
    assert m.game == index + 1
    assert m.pair == index // 2 + 1
    assert m.deal_seed == 20261461 + index // 2
    assert sorted((m.aware_seat, m.random_policy_seat,
                   m.random_opponent_seat)) == [0, 1, 2]
    assert m.margin == m.aware_score - m.random_policy_score
    assert m.h2h in ("AWARE", "RANDOM_POLICY", "TIE")
    assert m.outright in (
        "AWARE", "RANDOM_POLICY", "RANDOM_OPPONENT", "TIE_FOR_FIRST"
    )
    assert m.seconds > 0


def test_seat_swapped_match_uses_same_deals():
    a = play_match(0, seed=20261461, rollouts=8)
    b = play_match(1, seed=20261461, rollouts=8)
    assert a.deal_seed == b.deal_seed
    assert a.aware_seat == b.random_policy_seat
    assert a.random_policy_seat == b.aware_seat
    assert a.random_opponent_seat == b.random_opponent_seat


def test_reproducible_scores_across_parallel_game_workers(tmp_path):
    serial, se = tournament(4, seed=20261511, rollouts=8,
                            workers=1, csv_path=tmp_path / "serial.csv",
                            bootstrap=100)
    parallel, pe = tournament(4, seed=20261511, rollouts=8,
                              workers=2, csv_path=tmp_path / "parallel.csv",
                              bootstrap=100)
    assert [(m.game, m.deal_seed, m.aware_score, m.random_policy_score,
             m.random_opponent_score, m.margin, m.h2h, m.outright)
            for m in serial] == [
                (m.game, m.deal_seed, m.aware_score, m.random_policy_score,
                 m.random_opponent_score, m.margin, m.h2h, m.outright)
                for m in parallel
            ]
    assert se["average_margin"] == pe["average_margin"]
    assert se["bootstrap_95"] == pe["bootstrap_95"]
    assert (tmp_path / "serial.csv").read_text().splitlines()[0] == (
        tmp_path / "parallel.csv").read_text().splitlines()[0]


def test_paired_summary_rejects_invalid_games():
    m = play_match(0, seed=20261611, rollouts=8)
    with pytest.raises(ValueError, match="pairs"):
        summary([m])
    with pytest.raises(ValueError, match="pairs"):
        summary([m, replace(m, game=2)], bootstrap=100)
