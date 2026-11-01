"""Benchmark two-card policy cost and decision quality for 2-7 players.

Compares complete two-trick rollout times at identical fixed hands, bids and
random seeds. Decision-quality diagnostic uses the expected round score from
many *random continuation* samples on fixed hidden deals; this is not an
unbiased estimate of playing strength against strategic opponents.

Run:
  python -m benchmarks.endgame_policy --players 2 3 4 5 6 7 --states 1500
"""
import argparse
import random
import statistics
import time

from alphawhist.core.EndgamePolicy import two_card_values
from alphawhist.core.FastRollout import SimState, finish_round, legal_mask


def two_trick_states(nplayers, nstates, seed):
    """Make reproducible, synthetic legal two-card endgame situations."""
    rng = random.Random(seed + nplayers * 1009)
    out = []
    for _ in range(nstates):
        cards = rng.sample(range(52), nplayers * 2)
        hands = [sum(1 << c for c in cards[2*i:2*i+2])
                 for i in range(nplayers)]
        # Distribute five already-won tricks. The final two are still pending.
        won = [0]*nplayers
        for _ in range(5):
            won[rng.randrange(nplayers)] += 1
        bids = tuple(rng.randrange(8) for _ in range(nplayers))
        out.append(SimState(hands, [], bids, won, 0, 0, rng.randrange(4)))
    return out


def _copy(state):
    return SimState(state.hands.copy(), state.pile.copy(), state.bids,
                    state.tricks_won.copy(), state.leader, state.turn,
                    state.trump)


def throughput(states, policy, seed=7):
    rng = random.Random(seed)
    started = time.perf_counter()
    total = 0
    for base in states:
        scores = finish_round(_copy(base), rng, policy=policy)
        total += sum(scores)
    return time.perf_counter() - started, total


def quality_diagnostic(states, nrollouts=64, seed=17):
    """Compare approximate bid-aware action ranks to sampled random-play Q.

    An oracle using complete hidden deals here is an *evaluation reference*.
    The live policy never receives those hidden cards.
    """
    base_rng = random.Random(seed)
    comparisons = 0
    correct = 0
    greedy_gain = 0.0
    stochastic_gain = 0.0
    abs_error = 0.0
    for state in states:
        if state.hands[0].bit_count() != 2:
            continue
        values = two_card_values(state, legal_mask(state.hands[0], []))
        if values is None:
            continue
        a, b, va, vb = values
        estimates = []
        for card in (a, b):
            scores = 0
            for trial in range(nrollouts):
                sim = _copy(state)
                sim._play_legal(card)
                # Common random continuation seeds across the two actions.
                rng = random.Random(seed + 12971 * comparisons + trial)
                scores += finish_round(sim, rng, policy="random")[0]
            estimates.append(scores / nrollouts)
        qa, qb = estimates
        abs_error += abs(va + state.tricks_won[0] - qa) + abs(vb + state.tricks_won[0] - qb)
        chosen = a if va >= vb else b
        actual_best = a if qa >= qb else b
        correct += (chosen == actual_best)
        baseline = 0.5 * (qa + qb)
        greedy_gain += (qa if chosen == a else qb) - baseline
        prob_a = 1/(1 + __import__("math").exp((vb - va)/1.5))
        stochastic_gain += (prob_a*qa+(1-prob_a)*qb) - baseline
        comparisons += 1
    return {
        "decisions": comparisons,
        "best_action_match_pct": 100 * correct / comparisons if comparisons else 0,
        "greedy_gain_vs_uniform": greedy_gain / comparisons if comparisons else 0,
        "stochastic_gain_vs_uniform": stochastic_gain / comparisons if comparisons else 0,
        "value_mae_vs_random_rollout": abs_error / (2 * comparisons) if comparisons else 0,
    }


def run(players=(2, 3, 4, 5, 6, 7), states=1500, quality_states=40,
        quality_rollouts=64, seed=20261101):
    if states < 1 or quality_states < 1 or quality_rollouts < 1:
        raise ValueError("All sample sizes must be positive")
    if any(n not in range(2, 8) for n in players):
        raise ValueError("Player count must be between 2 and 7")
    results = []
    for n in players:
        situations = two_trick_states(n, states, seed)
        # Interleave repeat to reduce cold-start/timing-order bias.
        t_random, total_random = throughput(situations, "random", seed)
        t_aware, total_aware = throughput(situations, "bid_aware_2", seed)
        diagnostic = quality_diagnostic(
            situations[:min(quality_states, states)],
            nrollouts=quality_rollouts, seed=seed+n
        )
        row = {
            "players": n, "states": states,
            "random_seconds": t_random, "aware_seconds": t_aware,
            "slowdown": t_aware/t_random,
            "different_total_scores": total_random != total_aware,
            **diagnostic
        }
        print(
            f"BENCH players={n} endgames={states} "
            f"random_seconds={t_random:.4f} aware_seconds={t_aware:.4f} "
            f"slowdown={row['slowdown']:.3f}x "
            f"quality_decisions={diagnostic['decisions']} "
            f"best_action_match_pct={diagnostic['best_action_match_pct']:.2f} "
            f"greedy_gain_vs_uniform={diagnostic['greedy_gain_vs_uniform']:+.4f} "
            f"stochastic_gain_vs_uniform={diagnostic['stochastic_gain_vs_uniform']:+.4f} "
            f"value_mae_vs_random_rollout={diagnostic['value_mae_vs_random_rollout']:.4f}",
            flush=True
        )
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, nargs="+", default=list(range(2, 8)))
    parser.add_argument("--states", type=int, default=1500)
    parser.add_argument("--quality-states", type=int, default=40)
    parser.add_argument("--quality-rollouts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    run(args.players, args.states, args.quality_states,
        args.quality_rollouts, args.seed)


if __name__ == "__main__":
    main()
