"""Reproducible fast-vs-object-model rollout throughput benchmark.

Run from repository root:
    python -m benchmarks.rollouts --rollouts 100 --workers 2
No performance assertion: timings depend on runner hardware and load.
"""
import argparse
import contextlib
import io
import random
import time

from alphawhist.core.InformationState import InformationState
from alphawhist.core.FastRollout import evaluate_actions
from alphawhist.core.WhistStructure import Game


def make_state(cards=7):
    hand = tuple(["14h", "13h", "12s", "11c", "10d", "9s", "8c"][:cards])
    state = InformationState.new_round(
        hand, 3, round_index=0, dealer=2, leader=0, cards_per_player=cards
    )
    state = state.apply_public_event({
        "type": "trump", "player": 2, "suit": "h"
    })
    for player, bid in enumerate([1, 1, 0]):
        state = state.apply_public_event({
            "type": "bid", "player": player, "bid": bid
        })
    return state


def run_benchmark(rollouts=100, workers=1, cards=7):
    snapshot = make_state(cards)
    actions = snapshot.own_hand[:min(3, cards)]
    started = time.perf_counter()
    fast = evaluate_actions(
        snapshot, actions=actions, n_rollouts=rollouts,
        seed=314159, workers=workers, batch_size=64
    )
    fast_seconds = time.perf_counter() - started

    # Reference implementation: object graph + player objects + event log.
    # Uses the same snapshot and completes the same number of games per move.
    random.seed(314159)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in range(rollouts):
            for card in actions:
                game = Game.from_information_state(snapshot, seed=seed)
                game.playPartialTrick(card)
                game.playPartialRound()
                assert game.roundnumber == 1
    slow_seconds = time.perf_counter() - started
    count = len(actions) * rollouts
    print(f"cards={cards}, candidates={len(actions)}, worlds={rollouts}, "
          f"workers={workers}")
    print(f"Fast:   {count / fast_seconds:,.0f} complete rollouts/s "
          f"({fast_seconds:.3f}s)")
    print(f"Legacy: {count / slow_seconds:,.0f} complete rollouts/s "
          f"({slow_seconds:.3f}s)")
    print(f"Speedup (including setup): {slow_seconds / fast_seconds:.1f}x")
    assert fast.rollouts == rollouts
    return fast_seconds, slow_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cards", type=int, default=7, choices=range(1, 8))
    args = parser.parse_args()
    run_benchmark(args.rollouts, args.workers, args.cards)
