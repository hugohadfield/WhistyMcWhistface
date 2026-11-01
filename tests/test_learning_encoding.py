"""Observer-safe, fixed-width Whist neural state encoding."""
from dataclasses import replace
import random

import numpy as np
import pytest

from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistStructure import Game
from alphawhist.learning.encoding import (
    CARD_CHANNELS, CARD_SHAPE, CARD_TO_ID, FEATURE_DIM, GLOBAL_CHANNELS,
    GLOBAL_SHAPE, MAX_CARDS, MAX_PLAYERS, PLAYER_CHANNELS, PLAYER_SHAPE,
    encode_batch, encode_decision,
)


def started(own=("14h", "2s"), n=3, observer=0, leader=0):
    dealer = n - 1
    state = InformationState.new_round(
        own, n, round_index=3, dealer=dealer, leader=leader,
        cards_per_player=len(own), scores=[11 * i for i in range(n)],
        observer_id=observer,
    )
    state = state.apply_public_event({
        "type": "trump", "player": dealer, "suit": "h",
    })
    for pid in list(range(n)):
        state = state.apply_public_event({"type": "bid", "player": pid, "bid": 1})
    return state


def unpack(encoded):
    data = encoded.features
    csize = int(np.prod(CARD_SHAPE))
    psize = int(np.prod(PLAYER_SHAPE))
    return (data[:csize].reshape(CARD_SHAPE),
            data[csize:csize + psize].reshape(PLAYER_SHAPE),
            data[csize + psize:].reshape(GLOBAL_SHAPE))


@pytest.mark.parametrize("n", range(2, 8))
def test_shapes_dtypes_and_legal_actions_for_all_player_counts(n):
    cards = ("14h", "2s", "9c")
    state = started(cards, n=n)
    e = encode_decision(state)
    assert e.features.shape == (FEATURE_DIM,)
    assert e.features.dtype == np.float32
    assert e.legal_mask.dtype == np.bool_
    assert e.legal_mask.shape == (MAX_CARDS,)
    assert list(np.flatnonzero(e.legal_mask)) == sorted(CARD_TO_ID[c] for c in cards)
    assert e.remaining_tricks == 3
    card, player, global_features = unpack(e)
    assert np.all(card[:, CARD_CHANNELS.index("own")] == e.legal_mask.astype(float))
    assert player[:n, PLAYER_CHANNELS.index("active")].sum() == n
    assert not player[n:, :].any()
    assert global_features[GLOBAL_CHANNELS.index("num_players")] == pytest.approx(n/7)


def test_follow_suit_partial_trick_and_observer_relative_player_slots():
    state = started(("3h", "5s"), n=3, observer=1, leader=0)
    state = state.apply_public_event({"type": "card", "player": 0, "card": "9h"})
    assert state.turn == 1
    encoded = encode_decision(state)
    c, p, g = unpack(encoded)
    assert np.flatnonzero(encoded.legal_mask).tolist() == [CARD_TO_ID["3h"]]
    assert c[CARD_TO_ID["9h"], CARD_CHANNELS.index("played")] == 1
    assert c[CARD_TO_ID["9h"], CARD_CHANNELS.index("in_pile")] == 1
    assert c[CARD_TO_ID["9h"], CARD_CHANNELS.index("played_by_relative_2")] == 1
    assert c[CARD_TO_ID["9h"], CARD_CHANNELS.index("pile_by_relative_2")] == 1
    assert p[0, PLAYER_CHANNELS.index("self")] == 1
    assert p[2, PLAYER_CHANNELS.index("leader")] == 1
    assert g[GLOBAL_CHANNELS.index("lead_h")] == 1


def test_inferred_voids_are_visible_but_hands_are_not():
    state = started()
    state = state.apply_public_event({"type": "card", "player": 0, "card": "14h"})
    state = state.apply_public_event({"type": "card", "player": 1, "card": "3s"})
    state = state.apply_public_event({"type": "card", "player": 2, "card": "4h"})
    state = state.apply_public_event({"type": "trick_end", "winner": 0})
    assert state.turn == 0
    e = encode_decision(state)
    cards, players, _ = unpack(e)
    assert players[1, PLAYER_CHANNELS.index("void_h")] == 1
    assert cards[CARD_TO_ID["3s"], CARD_CHANNELS.index("played_by_relative_1")] == 1
    assert cards[CARD_TO_ID["2h"], CARD_CHANNELS.index("possible_for_relative_1")] == 0
    assert cards[CARD_TO_ID["2h"], CARD_CHANNELS.index("possible_for_relative_2")] == 1
    assert e.remaining_tricks == 1


def test_reordered_hand_and_policy_memory_do_not_change_features():
    state = started()
    reorder = replace(state, own_hand=tuple(reversed(state.own_hand)))
    sensitive_memory = state.with_policy_state({"private_generated_world": ["7h", "8h"]})
    reference = encode_decision(state)
    for other in (reorder, sensitive_memory):
        item = encode_decision(other)
        np.testing.assert_array_equal(item.features, reference.features)
        np.testing.assert_array_equal(item.legal_mask, reference.legal_mask)


def test_game_secret_opponent_hand_swap_cannot_change_encoding():
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [3]
    game.resetRound()
    random.seed(20263311)
    game.randomDeal(3)
    game.trumpsuit = "h"
    game._record_public_event({
        "type": "trump", "player": game.dealer, "suit": "h",
    })
    game.getPlayerBids()
    # The actor is seat 1 at round start; swap ONLY the other players.
    left, right = game.players[0], game.players[2]
    before = encode_decision(game.observe(1))
    a, b = left.real_hand, right.real_hand
    left.set_hands(b)
    right.set_hands(a)
    after = encode_decision(game.observe(1))
    np.testing.assert_array_equal(before.features, after.features)
    np.testing.assert_array_equal(before.legal_mask, after.legal_mask)


def test_batch_and_input_guards():
    state = started()
    batch = encode_batch([state, state])
    assert batch.features.shape == (2, FEATURE_DIM)
    assert batch.legal_mask.shape == (2, 52)
    assert batch.remaining_tricks.tolist() == [2, 2]
    assert batch.tricks_won.tolist() == [0, 0]
    assert batch.bids.tolist() == [1, 1]
    with pytest.raises(ValueError, match="empty"):
        encode_batch([])
    with pytest.raises(TypeError, match="InformationState"):
        encode_decision(object())
    with pytest.raises(ValueError, match="observer"):
        encode_decision(started(observer=1, leader=0))
    with pytest.raises(ValueError, match="hand size"):
        encode_decision(replace(state, trick_number=1))


def test_no_data_from_prior_rounds_beyond_explicit_current_match_score():
    state = started()
    prior = (
        '{"type":"round_start","round":2,"cards_per_player":2,"dealer":2,"leader":0}',
        '{"type":"card","player":2,"card":"6d"}',
    )
    other = replace(state, history_json=prior + state.history_json)
    assert not any(x["type"] == "card" for x in state.public_history)
    np.testing.assert_array_equal(encode_decision(state).features,
                                  encode_decision(other).features)
