"""Examine card-score convergence and move-choice stability vs rollout budget.

All budgets reuse the same seed, so the first N sampled worlds are identical
between small and large runs. The reference is a large *finite Monte Carlo
estimate*, not ground truth. No monotonic convergence is promised.

Examples:
    python -m benchmarks.score_convergence --scenario early
    python -m benchmarks.score_convergence --scenario mid --objective margin
    python -m benchmarks.score_convergence --snapshot decision_42.json \
        --rollouts 16 64 256 1024 4096 --reference 16384 --csv scores.csv
"""
import argparse
import csv
from pathlib import Path

from alphawhist.core.FastRollout import evaluate_actions
from alphawhist.core.InformationState import InformationState


SCENARIOS = {
    "early": (("10h", "14s", "2s", "11c", "5d", "6h", "8c"),
              (2, 1, 1)),
    "mid": (("10h", "14s", "2s"), (1, 1, 0)),
}


def example_snapshot(scenario):
    hand, bids = SCENARIOS[scenario]
    state = InformationState.new_round(
        hand, 3, round_index=0, dealer=2,
        leader=0, cards_per_player=len(hand),
    )
    state = state.apply_public_event(
        {"type": "trump", "player": 2, "suit": "h"}
    )
    for player, bid in enumerate(bids):
        state = state.apply_public_event(
            {"type": "bid", "player": player, "bid": bid}
        )
    return state


def convergence_rows(snapshot, budgets=(16, 64, 256, 1024, 4096),
                     reference_rollouts=8192, seed=42, objective="score",
                     workers=1):
    budgets = tuple(budgets)
    if (not budgets or any(n < 1 for n in budgets) or
            sorted(set(budgets)) != list(budgets)):
        raise ValueError("Rollout budgets must be distinct, positive and increasing")
    if reference_rollouts < budgets[-1]:
        raise ValueError("Reference budget must be at least the largest budget")

    ref = evaluate_actions(
        snapshot, n_rollouts=reference_rollouts, seed=seed,
        objective=objective, workers=workers,
    )
    ref_best = ref.best_action()
    ref_index = ref.actions.index(ref_best)
    rows = []
    summary = []
    for budget in budgets:
        result = evaluate_actions(
            snapshot, n_rollouts=budget, seed=seed, workers=workers,
            objective=objective,
        )
        if result.actions != ref.actions:
            raise AssertionError("Candidate action order changed")
        best = result.best_action()
        chosen_index = ref.actions.index(best)
        mae = sum(abs(v - r) for v, r in zip(result.values, ref.values)) / len(ref.actions)
        reference_regret = ref.values[ref_index] - ref.values[chosen_index]
        summary.append({
            "rollouts": budget,
            "best_card": best,
            "reference_best": ref_best,
            "mean_absolute_error": mae,
            "reference_regret": reference_regret,
        })
        for i, card in enumerate(result.actions):
            rows.append({
                "rollouts": budget,
                "card": card,
                "self_mean_points": result.mean_round_scores[i][snapshot.observer_id],
                "objective_value": result.values[i],
                "reference_value": ref.values[i],
                "error_vs_reference": result.values[i] - ref.values[i],
                "best_card_at_budget": best,
                "reference_best_card": ref_best,
                "reference_regret_of_best": reference_regret,
            })
    return summary, rows, ref


def run(snapshot, budgets, reference_rollouts, seed, objective, workers, csv_path=None):
    summary, rows, reference = convergence_rows(
        snapshot, budgets, reference_rollouts, seed, objective, workers
    )
    print(f"Observer={snapshot.observer_id}, cards={len(snapshot.own_hand)}, "
          f"seed={seed}, objective={objective}, reference={reference_rollouts}")
    print("Each point reuses the same initial seeded rollouts; reference is a finite estimate.")
    cards = reference.actions
    header = (f"{'N':>7}  " + " ".join(f"{card:>8}" for card in cards) +
              f"  {'chosen':>8}  {'MAE to ref':>10}  {'ref regret':>10}")
    print(header)
    for entry in summary:
        line = [row for row in rows if row["rollouts"] == entry["rollouts"]]
        values = " ".join(f"{row['objective_value']:8.3f}" for row in line)
        print(f"{entry['rollouts']:7d}  {values}  "
              f"{entry['best_card']:>8}  "
              f"{entry['mean_absolute_error']:10.3f}  "
              f"{entry['reference_regret']:10.3f}")
    print("Reference:", ", ".join(
        f"{card}={v:.3f}" for card, v in zip(reference.actions, reference.values)
    ), "best:", reference.best_action())
    if csv_path is not None:
        with Path(csv_path).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", csv_path)
    return summary, rows, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", choices=SCENARIOS, default="early")
    group.add_argument("--snapshot", type=Path, help="Saved observer snapshot JSON")
    parser.add_argument("--rollouts", type=int, nargs="+",
                        default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--reference", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--objective", choices=["score", "margin"], default="score")
    parser.add_argument("--csv", type=Path, help="Optional per-card CSV output")
    args = parser.parse_args()
    state = (InformationState.load(args.snapshot) if args.snapshot
             else example_snapshot(args.scenario))
    run(state, args.rollouts, args.reference, args.seed, args.objective,
        args.workers, args.csv)


if __name__ == "__main__":
    main()
