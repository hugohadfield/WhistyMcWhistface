"""Statistically meaningful Monte Carlo score convergence regressions.

A larger single sample is NOT guaranteed to be closer to the truth. Instead,
test a known analytic expectation across independent RNG seeds and verify
nested-prefix accounting exactly. The CLI experiment reports the full curves.
"""
from math import isclose
import random

import pytest

from alphawhist.core.FastRollout import _batch, evaluate_actions
from alphawhist.core.InformationState import InformationState


def make_snapshot(hand=("10h", "14s", "2s", "11c"), bids=(1, 1, 0),
                  nplayers=3, trump="h"):
    state = InformationState.new_round(
        hand, num_players=nplayers, round_index=0,
        dealer=nplayers - 1, leader=0, cards_per_player=len(hand),
    )
    state = state.apply_public_event(
        {"type": "trump", "player": nplayers - 1, "suit": trump}
    )
    for player, bid in enumerate(bids):
        state = state.apply_public_event(
            {"type": "bid", "player": player, "bid": bid}
        )
    return state


def test_prefix_consistency_for_card_scores_across_rollout_counts():
    """Small and large runs share the same first sampled worlds."""
    state = make_snapshot()
    actions = state.own_hand
    seed = 123456
    counts = (8, 32, 128, 512)
    for count in counts:
        observed = evaluate_actions(
            state, actions, n_rollouts=count, seed=seed, batch_size=17
        )
        numerator = _batch((state, actions, 0, count, seed, "random"))
        for index, row in enumerate(observed.mean_round_scores):
            for player_id, mean_score in enumerate(row):
                assert mean_score * count == pytest.approx(
                    numerator[index][player_id], abs=1e-9
                )
        assert observed.best_action() in actions

    # Prefix-additivity is exact for all players/actions, not merely average
    # convergence. It ensures score changes are due to additional rollouts.
    first = _batch((state, actions, 0, 32, seed, "random"))
    later = _batch((state, actions, 32, 128, seed, "random"))
    whole = _batch((state, actions, 0, 128, seed, "random"))
    assert whole == [[a + b for a, b in zip(x, y)]
                     for x, y in zip(first, later)]


@pytest.mark.parametrize("objective", ["score", "margin"])
def test_convergence_scores_reproducible_for_each_budget(objective):
    """Reruns and different batches do not change sampled card scores."""
    state = make_snapshot()
    for count in (8, 32, 128, 512):
        first = evaluate_actions(state, n_rollouts=count,
                                 seed=9991, objective=objective)
        second = evaluate_actions(state, n_rollouts=count,
                                  seed=9991, objective=objective,
                                  batch_size=13)
        assert first == second


def test_nontrivial_one_card_exact_expected_points_and_variance_decreases():
    """Ace of spades is beaten by any trump: P(win)=38/51.

    Two-player one-card game; we lead, our bid is one, and our payoff
    is 11 if the randomly dealt opponent lacks hearts, otherwise zero.
    Thus the exact expected score is 11*38/51. This is about the
    finite-sample mean, NOT a claim that all sample paths converge
    monotonically.
    """
    state = make_snapshot(
        hand=("14s",), bids=(1, 0), nplayers=2, trump="h"
    )
    exact = 11 * 38 / 51
    seeds = (1, 2, 3, 4, 5, 6, 7, 8)
    small = [
        evaluate_actions(state, n_rollouts=16, seed=seed).values[0]
        for seed in seeds
    ]
    large = [
        evaluate_actions(state, n_rollouts=4096, seed=seed).values[0]
        for seed in seeds
    ]
    small_mae = sum(abs(x - exact) for x in small) / len(seeds)
    large_mae = sum(abs(x - exact) for x in large) / len(seeds)
    # Aggregate across independent seeds for a robust convergence check.
    assert large_mae < small_mae / 2
    assert large_mae < 0.2
    assert abs(sum(large) / len(large) - exact) < 0.1


def test_multicard_estimates_stabilize_in_aggregate_not_every_sample():
    """Across several seeds, higher rollout budgets approach a large ref."""
    state = make_snapshot()
    ref = evaluate_actions(state, n_rollouts=32768, seed=771)
    seeds = (100, 200, 300, 400, 500, 600)
    errors = {}
    for count in (16, 1024):
        differences = []
        for seed in seeds:
            result = evaluate_actions(state, n_rollouts=count, seed=seed)
            differences.extend(abs(a - b)
                               for a, b in zip(result.values, ref.values))
        errors[count] = sum(differences) / len(differences)
    assert errors[1024] < errors[16] / 2


def test_convergence_does_not_mutate_observer_snapshot():
    state = make_snapshot()
    original = state.to_json()
    for count in (8, 32, 128, 512):
        evaluate_actions(state, n_rollouts=count, seed=88)
    assert state.to_json() == original


def test_different_objectives_share_identical_candidate_score_estimates():
    state = make_snapshot()
    for count in (16, 128, 512):
        own = evaluate_actions(state, n_rollouts=count, seed=25,
                               objective="score")
        relative = evaluate_actions(state, n_rollouts=count, seed=25,
                                    objective="margin")
        assert own.mean_round_scores == relative.mean_round_scores
        for row, diff in zip(relative.mean_round_scores, relative.values):
            assert isclose(diff, row[0] - max(row[1:]), abs_tol=1e-12)
