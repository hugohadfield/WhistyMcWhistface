"""Observer-state serialization, public inference and branch/rollout regressions."""
from dataclasses import FrozenInstanceError
from itertools import combinations
import json
import random

import pytest

from alphawhist.core.InformationState import InformationState, OpponentKnowledge
from alphawhist.core.WhistStructure import Game
from alphawhist.core.WhistLib import all_cards_in_deck


def fresh(hand=("14h", "2s"), n=3, dealer=2, leader=0):
    return InformationState.new_round(
        hand, n, 3, dealer, leader, len(hand), scores=[5] * n
    )


def started(hand=("14h", "2s"), n=3):
    s = fresh(hand, n)
    s = s.apply_public_event({"type": "trump", "player": 2, "suit": "h"})
    for pid in range(n):
        s = s.apply_public_event({"type": "bid", "player": pid, "bid": 0})
    return s


def append_card(s, pid, card):
    return s.apply_public_event({"type": "card", "player": pid, "card": card})


def test_starts_with_every_unknown_card_as_possibility():
    s = fresh()
    assert s.observer_id == 0
    assert s.own_hand == ("14h", "2s")
    assert all(p.remaining_count == 2 for p in s.opponents)
    assert all(len(p.possible_cards) == 50 for p in s.opponents)
    assert all(not set(s.own_hand) & p.possible_cards for p in s.opponents)


def test_history_contains_only_public_deal_parameters():
    s = fresh()
    assert s.public_history == ({
        "type": "round_start", "round": 3, "dealer": 2,
        "leader": 0, "cards_per_player": 2
    },)


def test_player_played_off_suit_removes_suit_from_beliefs():
    s = started()
    s = append_card(s, 0, "14h")
    s = append_card(s, 1, "3s")
    opp = next(p for p in s.opponents if p.player_id == 1)
    assert opp.remaining_count == 1
    assert "h" in opp.void_suits
    assert all(not c.endswith("h") for c in opp.possible_cards)
    assert "3s" not in opp.possible_cards
    assert "3s" not in s.opponents[1].possible_cards
    assert "14h" not in s.own_hand
    assert s.pile == ((0, "14h"), (1, "3s"))


def test_following_suit_does_not_create_void():
    s = append_card(started(), 0, "14h")
    s = append_card(s, 1, "3h")
    assert not s.opponents[0].void_suits


def test_every_played_card_removed_from_every_opponent():
    s = append_card(started(), 0, "14h")
    s = append_card(s, 1, "3s")
    assert all("3s" not in p.possible_cards for p in s.opponents)


def test_finishing_trick_updates_counts_and_leader():
    s = append_card(started(), 0, "14h")
    s = append_card(s, 1, "3s")
    s = append_card(s, 2, "4h")
    assert len(s.pile) == 3
    s = s.apply_public_event({"type": "trick_end", "winner": 0})
    assert s.tricks_won == (1, 0, 0)
    assert s.trick_number == 1
    assert s.leader == s.turn == 0
    assert s.pile == ()
    assert s.phase == "playing"


@pytest.mark.parametrize("event", [
    {"type": "card", "player": 1, "card": "3h"},
    {"type": "bid", "player": 1, "bid": 8},
    {"type": "trump", "player": 1, "suit": "s"},
])
def test_invalid_phase_or_action_rejected(event):
    with pytest.raises(ValueError):
        fresh().apply_public_event(event)


def test_cannot_play_observers_hand_as_opponent():
    s = append_card(started(), 0, "14h")
    with pytest.raises(ValueError):
        append_card(s, 1, "2s")


def test_own_hand_must_follow_suit():
    s = InformationState.new_round(
        ["14h", "2s"], 3, 0, 0, 1, 2
    )
    s = s.apply_public_event({"type": "trump", "player": 0, "suit": "s"})
    for pid in (1, 2, 0):
        s = s.apply_public_event({"type": "bid", "player": pid, "bid": 0})
    s = s.apply_public_event({"type": "card", "player": 1, "card": "3h"})
    s = s.apply_public_event({"type": "card", "player": 2, "card": "4h"})
    with pytest.raises(ValueError, match="follow suit"):
        s.apply_public_event({"type": "card", "player": 0, "card": "2s"})
    assert s.own_hand == ("14h", "2s")


def test_invalid_trick_winner_is_rejected():
    s = started()
    for pid, card in [(0, "14h"), (1, "3s"), (2, "4h")]:
        s = append_card(s, pid, card)
    with pytest.raises(ValueError, match="winner"):
        s.apply_public_event({"type": "trick_end", "winner": 1})


def test_frozen_and_immutable_nested_history():
    s = fresh().with_policy_state({"weights": [1, 2]})
    with pytest.raises(FrozenInstanceError):
        s.turn = 2
    memory = s.policy_state
    memory["weights"].append(999)
    history = s.public_history
    history[0]["dealer"] = 999
    assert s.policy_state == {"weights": [1, 2]}
    assert s.dealer == s.public_history[0]["dealer"] == 2


@pytest.mark.parametrize("phase", ["dealt", "bidding", "playing"])
def test_round_trip_every_phase(phase, tmp_path):
    s = fresh()
    if phase != "dealt":
        s = s.apply_public_event({"type": "trump", "player": 2, "suit": "h"})
    if phase == "playing":
        for pid in (0, 1, 2):
            s = s.apply_public_event({"type": "bid", "player": pid, "bid": 0})
        s = append_card(s, 0, "14h")
        s = append_card(s, 1, "3s")
    s = s.with_policy_state({"policy": "TestBot", "history": ["abc", 42]})
    encoded = s.to_json(sort_keys=True)
    assert InformationState.from_json(encoded) == s
    path = tmp_path / "snapshot.json"
    s.save(path)
    assert InformationState.load(path) == s
    assert InformationState.load(path).to_json() == s.to_json()


def test_snapshot_never_serializes_secret_hands():
    s = fresh().with_policy_state({"policy": "RandomPlayer"})
    raw = s.to_dict()
    assert "real_hand" not in json.dumps(raw)
    assert not any("real_hand" in p for p in raw["opponents"])


def test_from_dict_does_not_accept_extra_hidden_fields():
    raw = fresh().to_dict()
    raw["opponent_real_hands"] = [["3s", "4s"], ["5d", "6c"]]
    with pytest.raises(ValueError, match="schema"):
        InformationState.from_dict(raw)


def test_schema_version_must_match():
    raw = fresh().to_dict()
    raw["schema_version"] = 200
    with pytest.raises(ValueError, match="version"):
        InformationState.from_dict(raw)


def test_invalid_candidate_set_does_not_load():
    raw = fresh().to_dict()
    raw["opponents"][0]["possible_cards"].append("14h")
    with pytest.raises(ValueError):
        InformationState.from_dict(raw)


def test_no_duplicated_cards_and_candidate_constraints_after_sampling():
    s = started()
    s = append_card(s, 0, "14h")
    s = append_card(s, 1, "3s")
    for seed in range(100):
        hands = s.sample_hands(random.Random(seed))
        assert len(hands) == 3
        assert hands[0] == ("2s",)
        assert len(hands[1]) == 1 and len(hands[2]) == 2
        all_cards = [c for h in hands for c in h]
        assert len(all_cards) == len(set(all_cards))
        assert not set(all_cards) & set(s.played_cards)
        for opponent in s.opponents:
            assert set(hands[opponent.player_id]) <= opponent.possible_cards


def test_samples_do_not_change_snapshot():
    s = started()
    before = s.to_json()
    for seed in range(80):
        s.sample_hands(random.Random(seed))
    assert s.to_json() == before


def test_no_feasible_matching_raises_without_corrupting_state():
    s = started()
    opp = (
        OpponentKnowledge(1, 2, frozenset({"3s", "4s"})),
        OpponentKnowledge(2, 2, frozenset({"3s", "4s"})),
    )
    from dataclasses import replace
    impossible = replace(s, opponents=opp)
    with pytest.raises(ValueError, match="consistent"):
        impossible.sample_hands(random.Random(15))
    assert impossible.opponents == opp


def test_constraints_force_correct_unique_assignment():
    from dataclasses import replace
    s = fresh(("14h",), n=3)
    forced = replace(s, opponents=(
        OpponentKnowledge(1, 1, frozenset({"2h"})),
        OpponentKnowledge(2, 1, frozenset({"2h", "3s"})),
    ))
    for seed in range(50):
        assert forced.sample_hands(random.Random(seed)) == (
            ("14h",), ("2h",), ("3s",)
        )


def test_no_undealt_card_must_be_assigned():
    s = fresh(("14h",), n=3)
    hands = s.sample_hands(random.Random(2))
    assert sum(len(hand) for hand in hands) == 3  # not 52


def test_game_observation_does_not_depend_on_secret_opponent_hands():
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.resetRound()
    random.seed(123)
    game.randomDeal(2)
    before = game.observe()
    left, right = game.players[1], game.players[2]
    a, b = left.real_hand, right.real_hand
    left.set_hands(b, left.possible_hand)
    right.set_hands(a, right.possible_hand)
    assert game.observe() == before
    assert "real_hand" not in game.observe().to_json()


def test_observe_requires_deal_and_accepts_other_private_observers():
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    with pytest.raises(ValueError, match="Deal"):
        game.observe()
    game.resetRound()
    game.randomDeal(2)
    other = game.observe(player_id=1)
    assert other.observer_id == 1
    assert set(other.own_hand) == set(game.players[1].real_hand)
    assert not set(other.own_hand) & other.opponents[0].possible_cards
    with pytest.raises(ValueError, match="Invalid observer"):
        game.observe(player_id=5)


def setup_game(ncards=2, seed=234):
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [ncards]
    game.resetRound()
    random.seed(seed)
    game.randomDeal(ncards)
    game.trumpsuit = "h"
    game._record_public_event({"type": "trump", "player": game.dealer, "suit": "h"})
    game.getPlayerBids()
    game.turn = game.leader
    return game


def play_one_card(game):
    player = game.players[game.turn]
    card = player.getRealPossibleMoves(game.pile)[0]
    player.playCard(card)
    player.cardSuitCheck(game.pile, card)
    game.pile.append(card)
    game._record_public_event({
        "type": "card", "player": player.playerNumber, "card": card
    })
    game.cleanCardFromPlayers(card)
    game.incrementTurn()
    return card


@pytest.mark.parametrize("played", [0, 1, 2])
def test_mid_trick_snapshots_and_partial_continuation(played, tmp_path, capsys):
    game = setup_game(ncards=2)
    for _ in range(played):
        play_one_card(game)
    info = game.observe()
    path = tmp_path / "decision.json"
    info.save(path)
    del game
    loaded = InformationState.load(path)
    assert loaded == info
    for seed in range(10):
        rollout = Game.from_information_state(loaded, seed=seed)
        assert rollout.turn == loaded.turn
        assert rollout.pile == [card for _, card in loaded.pile]
        assert rollout.bids == list(loaded.bids)
        assert rollout.scores == list(loaded.scores)
        assert rollout.players[0].real_hand == list(loaded.own_hand)
        c = rollout.players[rollout.turn].getRealPossibleMoves(rollout.pile)[0]
        rollout.playPartialTrick(c)
        rollout.playPartialRound()
        assert rollout.roundnumber == loaded.round_index + 1
        assert sum(rollout.tricksWon) == 2


def test_multiple_branches_are_independent(capsys):
    game = setup_game(ncards=3)
    info = InformationState.from_json(game.observe().to_json())
    left = Game.from_information_state(info, seed=5)
    right = Game.from_information_state(info, seed=8)
    original = info.to_json()
    a = left.players[left.turn].getRealPossibleMoves(left.pile)[0]
    left.playPartialTrick(a)
    assert right.trickNumber == 0
    assert right._information_state == info
    assert info.to_json() == original


def test_public_state_progresses_through_full_game(capsys):
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    random.seed(44)
    for n in range(len(game.cardnumbers)):
        game.playFullRound()
        s = game.observe()
        assert s.round_index == n + 1
        assert s.phase == "between_rounds"
        assert s.scores == tuple(game.scores)
        assert all(b is not None for b in s.bids)
        assert len(s.own_hand) == 0
        assert s.trick_number == 0
        assert InformationState.from_json(s.to_json()) == s
    assert len(game.observe().public_history) > 100


def test_loaded_observation_continues_with_same_explicit_public_actions():
    s = started()
    s = append_card(s, 0, "14h")
    loaded = InformationState.from_json(s.to_json())
    for event in (
        {"type": "card", "player": 1, "card": "3s"},
        {"type": "card", "player": 2, "card": "4h"},
        {"type": "trick_end", "winner": 0},
    ):
        s = s.apply_public_event(event)
        loaded = loaded.apply_public_event(event)
        assert loaded == s


def test_serialized_public_history_can_be_read_by_new_algorithm():
    s = started()
    s = append_card(s, 0, "14h")
    s = append_card(s, 1, "3s")
    events = InformationState.from_json(s.to_json()).public_history
    assert [(e["player"], e["card"]) for e in events if e["type"] == "card"] == [
        (0, "14h"), (1, "3s")
    ]


def test_user_defined_policy_state_roundtrips():
    state = fresh().with_policy_state({
        "kind": "recurrent-policy-v1",
        "belief_logits": [-2.1, 0.4, 1.7],
        "nested": {"episode": 15, "mask": [True, False, None]},
    })
    assert InformationState.from_json(state.to_json()).policy_state == state.policy_state


def test_state_does_not_share_memory_with_from_dict_input():
    raw = fresh().to_dict()
    loaded = InformationState.from_dict(raw)
    raw["own_hand"].clear()
    raw["opponents"][0]["possible_cards"].clear()
    raw["public_history"].clear()
    assert loaded == fresh()


def test_invalid_own_hand_is_rejected():
    with pytest.raises(ValueError):
        InformationState.new_round(["2h", "2h"], 3, 0, 0, 1, 2)


def test_determinization_of_snapshot_not_tied_to_source_game(capsys):
    game = setup_game(ncards=2)
    state = game.observe()
    del game
    rollout = Game.from_information_state(state, seed=71)
    assert tuple(rollout.players[0].real_hand) == state.own_hand
    assert set(card for p in rollout.players for card in p.real_hand) <= set(all_cards_in_deck)


@pytest.mark.parametrize("hidden_field", [
    {"opponent_hand": ["5h", "6h"]},
    {"private_notes": "secret"},
    {"real_hand": ["2s"]},
])
def test_public_event_rejects_hidden_extra_fields(hidden_field):
    s = fresh()
    event = {"type": "trump", "player": 2, "suit": "h", **hidden_field}
    with pytest.raises(ValueError, match="schema"):
        s.apply_public_event(event)


def test_corrupted_history_with_hidden_card_field_rejected_on_load():
    raw = fresh().to_dict()
    raw["public_history"][0]["secret_opponent_hand"] = ["4h", "5h"]
    with pytest.raises(ValueError, match="schema"):
        InformationState.from_dict(raw)


def test_public_event_history_tracks_all_three_player_bids_in_order():
    s = started()
    assert [
        (e["player"], e["bid"])
        for e in s.public_history if e["type"] == "bid"
    ] == [(0, 0), (1, 0), (2, 0)]


def test_snapshot_pickle_not_required_and_independent_of_policy_class(tmp_path):
    state = fresh().with_policy_state({"policy": "custom-agent", "internal": {"x": 17}})
    path = tmp_path / "portable.json"
    state.save(path)
    source = path.read_text(encoding="utf-8")
    assert "custom-agent" in source
    assert "pickle" not in source.lower()
    assert InformationState.load(path).policy_state["internal"]["x"] == 17


def test_policy_export_and_restore_memory(capsys):
    game = setup_game(ncards=2)
    game.players[0].resetZeroCounter(1)
    snapshot = game.observe()
    assert snapshot.policy_state["zero_bid_count"] == 1
    restored = Game.from_information_state(snapshot, seed=1)
    assert restored.players[0].zero_bid_count == 1


def test_custom_policy_can_restore_its_own_persistent_state(capsys):
    from RandomPlayer import RandomPlayer

    class StatefulPlayer(RandomPlayer):
        def export_policy_state(self):
            return {"policy": "stateful", "zero_bid_count": self.zero_bid_count,
                    "cache": [1, 3, 5]}

        def load_policy_state(self, state):
            super().load_policy_state(state)
            self.cache = list(state["cache"])

    game = setup_game(ncards=2)
    game.players[0] = StatefulPlayer(0, game.players[0].real_hand)
    snapshot = game.observe()
    restored = Game.from_information_state(
        InformationState.from_json(snapshot.to_json()), seed=2,
        player_factory=lambda pid, hand: StatefulPlayer(pid, hand)
    )
    assert restored.players[0].cache == [1, 3, 5]


def test_other_observer_sees_only_public_cards_not_opponents_secrets():
    game = setup_game(ncards=3)
    before = game.observe(player_id=2)
    first = game.players[0].real_hand
    second = game.players[1].real_hand
    # These are both opponents from player 2's perspective.
    game.players[0].set_hands(second, game.players[0].possible_hand + [
        card for card in second if card not in game.players[0].possible_hand
    ])
    game.players[1].set_hands(first, game.players[1].possible_hand)
    after = game.observe(player_id=2)
    assert after == before


def test_observer_replay_preserves_void_suit_constraints_for_other_player():
    game = setup_game(ncards=2)
    first = game.observe(player_id=1)
    assert first.phase == "playing"
    assert len(first.opponents) == 2
    # The reconstructed history starts with the same public round start.
    assert first.public_history == game.observe(player_id=0).public_history
