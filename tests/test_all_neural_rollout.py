"""All-neural MC: multi-seat imperfect-information and gameplay regressions."""
import random
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphawhist.core.FastRollout import CARDS, CARD_ID, SimState, card_ids, legal_mask
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.fast_observation import FastObserver
from alphawhist.learning.encoding import encode_decision
from alphawhist.learning.checkpoint import save_checkpoint
from alphawhist.learning.model import DistributionalWhistMLP
from alphawhist.learning.neural_rollout import (
    _Branch, _advance, evaluate_neural_self,
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(812)
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    path = tmp_path_factory.mktemp("all-neural") / "tiny.pt"
    save_checkpoint(path, model, metadata={"teacher_policy": "bid_aware_2"})
    return path


def make_start(players, cards, seed, observer_id):
    rng = random.Random(seed)
    # Disjoint dealt original hands, not passed to perspective reconstruction.
    dealt = rng.sample(CARDS, players * cards)
    world = tuple(tuple(dealt[p*cards:(p+1)*cards])
                  for p in range(players))
    dealer = (observer_id - 1) % players
    leader = observer_id
    kwargs = dict(num_players=players, round_index=3,
                  dealer=dealer, leader=leader, cards_per_player=cards,
                  scores=tuple(20*i for i in range(players)))
    all_states = [
        InformationState.new_round(world[p], observer_id=p, **kwargs)
        for p in range(players)
    ]
    public_events = [{"type": "trump", "player": dealer, "suit": "s"}]
    public_events.extend(
        {"type": "bid", "player": (dealer + 1 + offset) % players,
         "bid": offset % (cards + 1)}
        for offset in range(players)
    )
    for event in public_events:
        all_states = [s.apply_public_event(event) for s in all_states]
    return world, all_states


def assert_equal_perspective(fast, reference):
    assert fast.observer_id == reference.observer_id
    assert fast.turn == reference.turn
    assert fast.leader == reference.leader
    assert fast.phase == reference.phase
    assert fast.trick_number == reference.trick_number
    assert tuple(fast.tricks_won) == reference.tricks_won
    assert set(CARDS[c] for c in card_ids(fast.own_hand)) == set(reference.own_hand)
    assert set(CARDS[c] for c in card_ids(fast.played)) == set(reference.played_cards)
    for opp in reference.opponents:
        pid = opp.player_id
        assert {CARDS[c] for c in card_ids(fast.possible[pid])} == set(opp.possible_cards)
        assert fast.remaining_counts[pid] == opp.remaining_count
        assert {suit for i, suit in enumerate("hcsd")
                if fast.void_suits[pid] & (1 << i)} == set(opp.void_suits)
    if fast.phase == "playing" and fast.turn == fast.observer_id:
        actual = fast.encode_decision()
        expected = encode_decision(reference)
        np.testing.assert_array_equal(actual.features, expected.features)
        np.testing.assert_array_equal(actual.legal_mask, expected.legal_mask)
        assert (actual.bid, actual.tricks_won, actual.remaining_tricks) == (
            expected.bid, expected.tricks_won, expected.remaining_tricks)


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("cards", [2, 4, 7])
def test_reconstructed_opponent_features_equal_authoritative_observer(
        players, cards):
    seed = 81715 + players * 89 + cards
    original, reference = make_start(players, cards, seed, observer_id=1)
    rng = random.Random(seed + 1)
    simulator = SimState.from_snapshot(reference[1], original)
    events = [json.loads(raw) for raw in reference[1].history_json]
    # Check at the start, after mid-trick plays, and after completed tricks.
    # Reconstruct ALL actors from their remaining sampled cards, never from
    # the rest of the sampled world.
    for step in range(players * cards):
        public = reference[1]
        assert public.phase == "playing"
        for pid in range(players):
            if pid == 1:
                fast = FastObserver.from_snapshot(public)
            else:
                own_remaining = tuple(CARDS[c] for c in card_ids(simulator.hands[pid]))
                fast = FastObserver.from_sampled_perspective(
                    public, pid, own_remaining, events)
            assert_equal_perspective(fast, reference[pid])
        legal = list(card_ids(legal_mask(simulator.hands[simulator.turn],
                                          simulator.pile)))
        choice = rng.choice(legal)
        event = {"type": "card", "player": simulator.turn,
                 "card": CARDS[choice]}
        previous_pile_size = len(simulator.pile)
        simulator._play_legal(choice)
        reference = [s.apply_public_event(event) for s in reference]
        events.append(event)
        if not simulator.pile:
            ended = {"type": "trick_end", "winner": simulator.leader}
            reference = [s.apply_public_event(ended) for s in reference]
            events.append(ended)
    assert reference[1].phase == "scoring"


def test_opponent_features_do_not_depend_on_other_sampled_hands():
    world, states = make_start(3, 3, 7118, observer_id=0)
    snapshot = states[0]
    events = [json.loads(raw) for raw in snapshot.history_json]
    a = FastObserver.from_sampled_perspective(snapshot, 1, world[1], events)
    b = FastObserver.from_sampled_perspective(
        snapshot, 1, tuple(reversed(world[1])), events)
    assert a == b
    assert not hasattr(a, "hands")
    assert not hasattr(a, "sampled_hands")
    # The third player's specific sampled hand is NEVER supplied.
    assert a.possible[2] == b.possible[2]
    assert a.possible[2] != sum(1 << CARD_ID[c] for c in world[2])


def test_all_neural_continuations_are_deterministic_across_batch_sizes(checkpoint):
    world, states = make_start(3, 4, 91573, observer_id=0)
    snapshot = states[0]
    kw = dict(checkpoint=checkpoint, opponent_policy="neural",
              n_rollouts=9, seed=1992, workers=1,
              collect_trick_histograms=True)
    a = evaluate_neural_self(snapshot, worlds_per_batch=2, batch_size=3, **kw)
    b = evaluate_neural_self(snapshot, worlds_per_batch=4, batch_size=8, **kw)
    assert a == b
    assert len(a.actions) == 4
    for counts in a.additional_trick_counts:
        assert sum(counts) == 9
    # Replay the same sampled world with ALL perspectives: no illegal action
    # can be accepted by any observer or the simulation.
    sampled = snapshot.sample_hands(random.Random(1992))
    base = SimState.from_snapshot(snapshot, sampled)
    templates = tuple(
        FastObserver.from_snapshot(snapshot) if pid == 0 else
        FastObserver.from_sampled_perspective(
            snapshot, pid, sampled[pid],
            [json.loads(raw) for raw in snapshot.history_json])
        for pid in range(3))
    all_obs = tuple(v.fork() for v in templates)
    branch = _Branch(base, all_obs[0], random.Random(5), 0, all_obs)
    _advance(branch, CARD_ID[snapshot.own_hand[0]])
    assert all(v.turn == branch.state.turn for v in all_obs)
    assert all(v.played == all_obs[0].played for v in all_obs)


def test_neural_opponent_policy_rejected_without_valid_checkpoint(tmp_path):
    world, states = make_start(2, 2, 7731, observer_id=0)
    with pytest.raises(FileNotFoundError):
        evaluate_neural_self(states[0], checkpoint=tmp_path / "missing.pt",
                             opponent_policy="neural")
