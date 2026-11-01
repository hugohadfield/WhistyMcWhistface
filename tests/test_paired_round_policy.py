"""Matched-round A/B benchmark: fair deals, reproducibility and CSV safety."""
import csv
from dataclasses import replace

import pytest

from benchmarks.paired_round_policy import (
    bootstrap_ci, run_scenario, summarize, tournament,
)


@pytest.mark.parametrize("players,cards", [
    (2, 1), (3, 2), (3, 3), (3, 7), (5, 4), (7, 7),
])
def test_identical_policy_is_exactly_reproducible(players, cards):
    record = run_scenario(
        index=1, cards=cards, seed=20261119, rollouts=8,
        players=players, policy_a="random", policy_b="random",
    )
    assert record.score_a == record.score_b
    assert record.tricks_a == record.tricks_b
    assert record.margin_delta == record.score_delta == 0
    assert record.exact_bid_a == record.exact_bid_b
    assert record.players == players and record.cards == cards
    assert len(record.deal_sha256) == 64


@pytest.mark.parametrize("cards", [1, 2, 3, 5, 7])
def test_bid_aware_comparison_replays_exact_round(cards):
    row = run_scenario(3, cards, seed=20261123, rollouts=8)
    assert row.policy_a == "random"
    assert row.policy_b == "bid_aware_2"
    assert row.tested_seat == 0
    assert row.score_delta == row.score_b - row.score_a
    assert row.margin_delta == row.margin_b - row.margin_a
    assert row.tricks_a <= cards and row.tricks_b <= cards
    assert 0 <= row.score_a <= cards + 10
    assert 0 <= row.score_b <= cards + 10
    assert row.exact_bid_a in (0, 1) and row.exact_bid_b in (0, 1)


def test_one_card_round_is_a_negative_control():
    results = [
        run_scenario(i, 1, seed=20261203, rollouts=8)
        for i in range(6)
    ]
    assert all(x.score_delta == 0 for x in results)
    assert all(x.margin_delta == 0 for x in results)


def test_same_seed_worker_counts_and_streaming_csv(tmp_path):
    first_csv = tmp_path / "serial.csv"
    second_csv = tmp_path / "parallel.csv"
    kwargs = dict(
        rounds_per_size=3, cards=(1, 2, 3), rollouts=8, players=3,
        seed=20261209, bootstrap=200, progress_every=2,
    )
    serial, summary_serial = tournament(
        **kwargs, workers=1, csv_path=first_csv,
    )
    parallel, summary_parallel = tournament(
        **kwargs, workers=2, csv_path=second_csv,
    )
    strip_time = lambda rows: [replace(r, seconds=0) for r in rows]
    assert strip_time(serial) == strip_time(parallel)
    assert summary_serial == summary_parallel
    with first_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert rows[0]["score_delta"] == str(serial[0].score_delta)
    assert rows[-1]["deal_seed"] == str(serial[-1].deal_seed)
    assert first_csv.read_text().splitlines()[0] == (
        second_csv.read_text().splitlines()[0]
    )


def test_summary_bootstrap_and_arguments():
    runs = [run_scenario(i, 2, seed=20261221, rollouts=8)
            for i in range(4)]
    stats = summarize(runs, bootstrap=200, seed=123)
    assert stats[0]["cards"] == 0
    assert stats[1]["cards"] == 2
    assert stats[0]["rounds"] == 4
    assert stats[0]["wins"] + stats[0]["losses"] + stats[0]["ties"] == 4
    assert bootstrap_ci([2, 2, 2], seed=12, iterations=200) == (2, 2)
    with pytest.raises(ValueError):
        run_scenario(0, 8, rollouts=8)
    with pytest.raises(ValueError):
        run_scenario(0, 7, players=8, rollouts=8)
    with pytest.raises(ValueError):
        tournament(1, cards=(1, 1), rollouts=8, bootstrap=100)
    with pytest.raises(ValueError):
        tournament(1, cards=(7,), players=8, rollouts=8, bootstrap=100)


@pytest.mark.parametrize("cards", [2, 4])
def test_same_policy_is_identical_against_monte_carlo_opponents(cards):
    row = run_scenario(
        index=5, cards=cards, seed=20262211, rollouts=8,
        opponents="monte_carlo", policy_a="bid_aware_2",
        policy_b="bid_aware_2",
    )
    assert row.opponents == "monte_carlo"
    assert row.score_delta == 0
    assert row.margin_delta == 0
    assert row.tricks_a == row.tricks_b


def test_strong_opponents_matched_deal_reproducible_across_workers(tmp_path):
    kwargs = dict(
        rounds_per_size=2, cards=(3, 4), rollouts=8, players=3,
        seed=20262219, bootstrap=200, progress_every=2,
        opponents="monte_carlo",
    )
    serial, serial_summary = tournament(
        **kwargs, workers=1, csv_path=tmp_path / "one.csv",
    )
    parallel, parallel_summary = tournament(
        **kwargs, workers=2, csv_path=tmp_path / "two.csv",
    )
    assert [replace(row, seconds=0) for row in serial] == [
        replace(row, seconds=0) for row in parallel
    ]
    assert serial_summary == parallel_summary
    assert all(row.opponents == "monte_carlo" for row in serial)


def test_invalid_opponent_type_is_rejected():
    with pytest.raises(ValueError, match="opponent"):
        run_scenario(0, 3, rollouts=8, opponents="unknown")
    with pytest.raises(ValueError, match="opponent"):
        tournament(1, cards=(3,), rollouts=8, opponents="unknown")
