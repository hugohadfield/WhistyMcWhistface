"""Exact feature equivalence between fast and authoritative rollout observers."""
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphawhist.core.FastRollout import (
    CARD_ID, CARDS, SimState, _mix, card_ids, legal_mask,
)
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.checkpoint import save_checkpoint
from alphawhist.learning.encoding import encode_decision
from alphawhist.learning.fast_observation import FastObserver
from alphawhist.learning.model import DistributionalWhistMLP
from alphawhist.learning.neural_rollout import (
    _Branch, _advance, _finish_group, _load_model, evaluate_neural_self,
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(9001)
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    checkpoint = tmp_path_factory.mktemp("fast-observer") / "frozen.pt"
    save_checkpoint(checkpoint, model, metadata={"teacher_policy": "bid_aware_2"})
    return checkpoint


def initial_information(n, size, me, seed):
    rng = random.Random(seed)
    own = tuple(rng.sample(CARDS, size))
    dealer = seed % n
    leader = (dealer + 1) % n
    # Include a prior round in the public log, which must NOT enter current
    # round played-by features. The neural encoding scans to round_start.
    earlier = (json.dumps({
        "type": "round_start", "round": 0, "cards_per_player": 1,
        "dealer": dealer, "leader": leader,
    }), json.dumps({"type": "card", "player": me, "card": "14d"}))
    snap = InformationState.new_round(
        own, num_players=n, round_index=4, dealer=dealer,
        leader=leader, cards_per_player=size, observer_id=me,
        previous_history=earlier, scores=tuple(range(0, 20*n, 20)),
    )
    snap = snap.apply_public_event(
        {"type": "trump", "player": dealer, "suit": "s"})
    for step in range(n):
        player = (dealer + 1 + step) % n
        snap = snap.apply_public_event({
            "type": "bid", "player": player, "bid": step % (size + 1)})
    return snap


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("cards", [2, 4, 7])
def test_public_history_and_features_identical_after_every_simulated_card(
        players, cards):
    seed = 11700 + players * 37 + cards
    me = seed % players
    snapshot = initial_information(players, cards, me, seed)
    hand_rng = random.Random(seed + 1)
    world = snapshot.sample_hands(hand_rng)
    original = SimState.from_snapshot(snapshot, world)
    fast = _Branch(
        SimState(original.hands.copy(), original.pile.copy(), original.bids,
                 original.tricks_won.copy(), original.leader, original.turn,
                 original.trump),
        FastObserver.from_snapshot(snapshot), random.Random(seed + 2), 0,
    )
    reference = _Branch(
        original, snapshot, random.Random(seed + 2), 0,
    )
    assert snapshot.phase == "playing"
    choices = random.Random(seed + 3)
    comparisons = 0
    for _ in range(cards * players):
        old = reference.observation
        f = fast.observation
        assert old.turn == f.turn
        assert old.leader == f.leader
        assert old.trick_number == f.trick_number
        assert old.tricks_won == tuple(f.tricks_won)
        assert old.phase == f.phase
        assert tuple(old.bids) == tuple(f.bids)
        assert set(old.own_hand) == {
            CARDS[c] for c in card_ids(f.own_hand)}
        assert set(old.played_cards) == {
            CARDS[c] for c in card_ids(f.played)}
        for opp in old.opponents:
            assert set(opp.possible_cards) == {
                CARDS[c] for c in card_ids(f.possible[opp.player_id])}
            assert opp.remaining_count == f.remaining_counts[opp.player_id]
            assert set(opp.void_suits) == {
                "hcsd"[s] for s in range(4)
                if f.void_suits[opp.player_id] & (1 << s)}
        if old.turn == me and old.phase == "playing":
            expected = encode_decision(old)
            actual = f.encode_decision()
            np.testing.assert_array_equal(actual.features, expected.features)
            np.testing.assert_array_equal(actual.legal_mask, expected.legal_mask)
            assert (actual.remaining_tricks, actual.tricks_won, actual.bid) == (
                expected.remaining_tricks, expected.tricks_won, expected.bid)
            comparisons += 1
        legal = legal_mask(original.hands[original.turn], original.pile)
        selected = choices.choice(list(card_ids(legal)))
        _advance(reference, selected)
        _advance(fast, selected)
        assert reference.state.tricks_won == fast.state.tricks_won
        assert reference.state.hands == fast.state.hands
    assert comparisons >= 1
    assert fast.observation.phase == reference.observation.phase == "scoring"


def test_fast_observer_cannot_see_hidden_worlds():
    snapshot = initial_information(3, 3, 0, 233)
    a = FastObserver.from_snapshot(snapshot)
    b = FastObserver.from_snapshot(snapshot)
    assert a.fork() == b.fork()
    # A public card by another actor updates beliefs, not private hand labels.
    # The fast observation carries no sampled complete deal.
    assert not hasattr(a, "hands")
    assert not hasattr(a, "sampled_hands")
    assert a.possible == b.possible


def reference_rollouts(snapshot, checkpoint, n_rollouts, seed):
    actions = tuple(snapshot.own_hand)
    sums = [[0] * snapshot.num_players for _ in actions]
    hist = [[0] * 8 for _ in actions]
    model = _load_model(str(checkpoint.resolve()))
    deal_rng = random.Random()
    for index in range(n_rollouts):
        deal_rng.seed(_mix(seed, 2 * index))
        world = snapshot.sample_hands(deal_rng)
        base = SimState.from_snapshot(snapshot, world)
        branches = []
        seed_for_policy = _mix(seed, 2 * index + 1)
        for i, action in enumerate(actions):
            branch = _Branch(
                SimState(base.hands.copy(), base.pile.copy(), base.bids,
                         base.tricks_won.copy(), base.leader, base.turn,
                         base.trump),
                snapshot,
                random.Random(seed_for_policy), i,
            )
            _advance(branch, CARD_ID[action])
            branches.append(branch)
        _finish_group(branches, model, "bid_aware_2", sums, hist,
                      snapshot.tricks_won[snapshot.observer_id])
    return sums, hist


def test_fast_and_reference_rollouts_produce_bit_identical_training_targets(
        checkpoint):
    # Controlled RNG and frozen net; changing observation encoding must not
    # change any sampled world, continuation policy or resulting histogram.
    snapshot = initial_information(3, 4, 0, 1173)
    assert snapshot.turn == snapshot.observer_id  # choose 1173%3=0 as dealer
    seed, count = 66512, 13
    old_sums, old_counts = reference_rollouts(snapshot, checkpoint, count, seed)
    new = evaluate_neural_self(
        snapshot, checkpoint=checkpoint, n_rollouts=count,
        seed=seed, workers=1, batch_size=7, worlds_per_batch=3,
        collect_trick_histograms=True,
    )
    assert new.actions == tuple(CARDS[i] for i in sorted(
        CARD_ID[card] for card in snapshot.own_hand))
    by_action = {card: i for i, card in enumerate(snapshot.own_hand)}
    for i, card in enumerate(new.actions):
        assert new.additional_trick_counts[i] == tuple(old_counts[by_action[card]])
        expected = tuple(row / count for row in old_sums[by_action[card]])
        assert new.mean_round_scores[i] == expected
