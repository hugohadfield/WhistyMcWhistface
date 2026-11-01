"""Large paired-reference diagnostic on *legal, played* 2-card endgames.

Each independent state is sampled from a complete 2..7-player RandomPlayer
round of 2..7 cards: real deals, legal trump selection, valid ordered bids,
complete legal play up to the final two tricks, and public card/void history.
Only ONE decision state is retained per independent round seed.

Targets: evaluate both legal first cards using identical hidden-world samples
and independent, common-random-number opponent continuations. Reference
opponents play uniformly at random (NOT an optimal-play oracle).

Run (one player count at a time; matrix jobs recommended):
  python -m benchmarks.endgame_quality --players 3 --states 1000 --rollouts 2048 --csv quality-3.csv
"""
import argparse
import contextlib
import csv
import math
import os
from pathlib import Path
import random
import statistics
import time

from RandomPlayer import RandomPlayer
from alphawhist.core.EndgamePolicy import _TEMPERATURE, two_card_values
from alphawhist.core.FastRollout import (
    CARD_ID, SUITS, SimState, _mix, finish_round, hand_mask, legal_mask,
)
from alphawhist.core.WhistStructure import Game


class CollectingRandomPlayer(RandomPlayer):
    """Capture legitimate in-game information states, never private rival cards."""

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        possible = self.getRealPossibleMoves(pile)
        if len(self.real_hand) == 2 and len(possible) == 2:
            snap = fullGameObject.observe(self.playerNumber)
            assert snap.phase == "playing" and snap.turn == snap.observer_id
            assert len(snap.own_hand) == 2
            fullGameObject._endgame_snapshots.append(snap)
        return super().makeMove(pile, trumpsuit, fullGameObject)


def collect_state(nplayers, seed, cards_per_player=None):
    """Run one real legal round; pick one decision uniformly from its final two tricks.

    A fresh game, dealt cards, and RNG seed are used for each state. The whole
    round completes, so no partial or invented bids/trick counts are created.
    """
    if not 2 <= nplayers <= 7:
        raise ValueError("Number of players must be 2..7")
    rng = random.Random(seed ^ 0x95E2B409)
    size = cards_per_player if cards_per_player is not None else rng.randrange(2, 8)
    if not 2 <= size <= 7:
        raise ValueError("Round size must be 2..7")
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.players = [CollectingRandomPlayer(i) for i in range(nplayers)]
    game.numberofplayers = nplayers
    game.cardnumbers = [size]
    game.resetGame()
    game._endgame_snapshots = []
    random.seed(seed)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            game.playFullRound()
    assert game.roundnumber == 1
    assert sum(game.tricksWon) == size
    assert game._endgame_snapshots
    snap = rng.choice(game._endgame_snapshots)
    assert snap.num_players == nplayers and snap.observer_id == snap.turn
    assert len(snap.own_hand) == 2
    return snap


def heuristics(snapshot):
    """Compute first-card preference without an opponent-hand determinization."""
    n = snapshot.num_players
    observer = snapshot.observer_id
    hands = [0] * n
    hands[observer] = hand_mask(snapshot.own_hand)
    state = SimState(
        hands, [CARD_ID[card] for _, card in snapshot.pile],
        tuple(snapshot.bids), list(snapshot.tricks_won),
        snapshot.leader, snapshot.turn, SUITS.index(snapshot.trump)
    )
    mask = legal_mask(hands[observer], state.pile)
    values = two_card_values(state, mask)
    if values is None:
        raise ValueError("Expected two legal moves")
    a, b, va, vb = values
    # Score a 'high/low' bidder baseline on public card strength, with
    # direction set by deficit. This makes no hidden-card assumptions.
    def strength(card):
        suit = card // 13
        lead = state.pile[0] // 13 if state.pile else None
        return (100 if suit == state.trump else
                10 if lead is not None and suit == lead else 0) + card % 13
    need = snapshot.bids[observer] - snapshot.tricks_won[observer]
    # Need 0 or 1: keep a high card for next trick; need 2: attack now.
    highlow = a if ((strength(a) >= strength(b)) == (need >= 2)) else b
    greedy = a if va >= vb else b
    probability_a = 1 / (1 + math.exp((vb - va) / _TEMPERATURE))
    return (a, b, va, vb, greedy, highlow, probability_a)


def paired_reference(snapshot, a, b, rollouts=2048, seed=0):
    """Same independent hidden worlds and seeded random continuations for a,b.

    Each world is drawn only from the acting player's public information; the
    opponent hands are NOT exposed to heuristics. Paired per-world differences
    permit a standard error of the estimated action-value *difference*.
    """
    if rollouts < 2:
        raise ValueError("Reference requires at least two rollouts")
    dealt_rng = random.Random()
    future_rng = random.Random()
    actions = (a, b)
    totals = [0, 0]
    deltas_sum = 0.0
    deltas_sum_sq = 0.0
    for index in range(rollouts):
        dealt_rng.seed(_mix(seed, 2 * index))
        world = snapshot.sample_hands(dealt_rng)
        base = SimState.from_snapshot(snapshot, world)
        continuation_seed = _mix(seed, 2 * index + 1)
        scores = [0, 0]
        for j, card in enumerate(actions):
            state = SimState(base.hands.copy(), base.pile.copy(), base.bids,
                             base.tricks_won.copy(), base.leader, base.turn,
                             base.trump)
            state.play(card)
            future_rng.seed(continuation_seed)
            scores[j] = finish_round(state, future_rng, "random")[snapshot.observer_id]
            totals[j] += scores[j]
        delta = scores[0] - scores[1]
        deltas_sum += delta
        deltas_sum_sq += delta * delta
    qa, qb = totals[0] / rollouts, totals[1] / rollouts
    mean_diff = deltas_sum / rollouts
    variance = max(0.0, (deltas_sum_sq - rollouts * mean_diff**2) / (rollouts - 1))
    se_diff = math.sqrt(variance / rollouts)
    return qa, qb, se_diff


def _summary(rows):
    """Descriptive per-independent-round statistics; no binomial cherry-pick."""
    n = len(rows)
    if not n:
        raise ValueError("No observations")
    keys = ["regret_bid_aware", "regret_softmax", "regret_highlow",
            "regret_random", "score_gain_softmax_vs_random",
            "score_gain_greedy_vs_random", "score_gain_highlow_vs_random"]
    result = {"states": n}
    for key in keys:
        data = [r[key] for r in rows]
        mean = statistics.mean(data)
        se = statistics.stdev(data) / math.sqrt(n) if n > 1 else 0.0
        result[key] = mean
        result[key + "_ci95"] = (mean - 1.96 * se, mean + 1.96 * se)
    result["agreement"] = sum(r["agrees"] for r in rows) / n
    result["clear_states"] = sum(r["clear_gap"] for r in rows)
    obvious = [r for r in rows if r["clear_gap"]]
    result["clear_agreement"] = (
        sum(r["agrees"] for r in obvious) / len(obvious) if obvious else None)
    result["ambiguous_below_0_5"] = sum(r["gap"] < 0.5 for r in rows)
    result["gap_ge_1"] = sum(r["gap"] >= 1 for r in rows)
    result["gap_ge_2"] = sum(r["gap"] >= 2 for r in rows)
    return result


def run(players=3, states=1000, rollouts=2048, seed=20261101, csv_path=None,
        progress=100):
    if not 2 <= players <= 7 or states < 1 or rollouts < 2 or progress < 1:
        raise ValueError("Invalid player count or evaluation size")
    outputs = []
    started = time.perf_counter()
    for i in range(states):
        # Each unique seed generates an independent fully completed legal
        # game. Vary all 2..7 round sizes deterministically across cases.
        game_seed = seed + players * 1000009 + i * 1009
        snapshot = collect_state(players, game_seed, 2 + (i % 6))
        a, b, va, vb, greedy, highlow, pa = heuristics(snapshot)
        qa, qb, se = paired_reference(
            snapshot, a, b, rollouts, seed=game_seed ^ 0xEDE123A1
        )
        better = a if qa >= qb else b
        best = max(qa, qb)
        greedy_q = qa if greedy == a else qb
        highlow_q = qa if highlow == a else qb
        soft_q = pa * qa + (1 - pa) * qb
        random_q = (qa + qb) / 2
        delta = qa - qb
        row = {
            "players": players, "case": i, "round_seed": game_seed,
            "cards_in_round": snapshot.cards_per_player,
            "observer": snapshot.observer_id, "pile_size": len(snapshot.pile),
            "bid": snapshot.bids[snapshot.observer_id],
            "tricks_won": snapshot.tricks_won[snapshot.observer_id],
            "action_a": a, "action_b": b,
            "heuristic_a": va, "heuristic_b": vb,
            "reference_a": qa, "reference_b": qb, "reference_diff_se": se,
            "reference_gap": abs(delta), "gap": abs(delta),
            "clear_gap": abs(delta) > 1.96 * se,
            "agrees": greedy == better,
            "greedy_action": greedy, "highlow_action": highlow,
            "probability_a": pa,
            "regret_bid_aware": best - greedy_q,
            "regret_softmax": best - soft_q,
            "regret_highlow": best - highlow_q,
            "regret_random": best - random_q,
            "score_gain_softmax_vs_random": soft_q - random_q,
            "score_gain_greedy_vs_random": greedy_q - random_q,
            "score_gain_highlow_vs_random": highlow_q - random_q,
        }
        outputs.append(row)
        if (i + 1) % progress == 0:
            print(f"PROGRESS players={players} states={i+1} "
                  f"elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    total_seconds = time.perf_counter()-started
    stats = _summary(outputs)
    def interval(k):
        lo, hi = stats[k + "_ci95"]
        return f"{stats[k]:+.4f}[{lo:+.4f},{hi:+.4f}]"
    print(
        f"QUALITY players={players} states={states} rollouts={rollouts} "
        f"agreement_pct={100*stats['agreement']:.2f} "
        f"clear_states={stats['clear_states']} "
        f"clear_agreement_pct={100*stats['clear_agreement']:.2f}" if
        stats["clear_agreement"] is not None else
        f"QUALITY players={players} states={states} rollouts={rollouts} "
        f"agreement_pct={100*stats['agreement']:.2f} "
        f"clear_states=0 clear_agreement_pct=NA",
        flush=True
    )
    print(
        f"REGRET players={players} "
        f"greedy={interval('regret_bid_aware')} "
        f"softmax={interval('regret_softmax')} "
        f"highlow={interval('regret_highlow')} "
        f"random={interval('regret_random')} "
        f"softmax_gain={interval('score_gain_softmax_vs_random')} "
        f"greedy_gain={interval('score_gain_greedy_vs_random')} "
        f"highlow_gain={interval('score_gain_highlow_vs_random')} "
        f"ambiguous={stats['ambiguous_below_0_5']} "
        f"gap_ge_1={stats['gap_ge_1']} gap_ge_2={stats['gap_ge_2']} "
        f"wall_s={total_seconds:.2f}", flush=True
    )
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=outputs[0].keys())
            writer.writeheader()
            writer.writerows(outputs)
    return outputs, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=range(2, 8), default=3)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    run(args.players, args.states, args.rollouts, args.seed, args.csv,
        progress=args.progress)


if __name__ == "__main__":
    main()
