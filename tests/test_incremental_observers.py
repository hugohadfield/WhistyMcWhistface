"""Cached Game.observe must be strictly equivalent to public-event replay."""
import contextlib
import io
import json
import random

from alphawhist.core.WhistStructure import Game
from alphawhist.core.InformationState import InformationState


def _reference_observe(game, player_id):
    """Pre-cache reconstruction, independent of the production fast path."""
    root = game._information_state
    events = root.public_history
    first = max(i for i, event in enumerate(events)
                if event["type"] == "round_start")
    round_events = events[first:]
    beginning = round_events[0]
    starting_hand = list(game.players[player_id].real_hand)
    starting_hand.extend(
        event["card"] for event in round_events[1:]
        if event["type"] == "card" and event["player"] == player_id
    )
    state = InformationState.new_round(
        own_hand=starting_hand,
        num_players=game.numberofplayers,
        round_index=beginning["round"],
        dealer=beginning["dealer"], leader=beginning["leader"],
        cards_per_player=beginning["cards_per_player"],
        scores=game.scores, observer_id=player_id,
        previous_history=root.history_json[:first],
    )
    for event in round_events[1:]:
        state = state.apply_public_event(event)
    return state.with_policy_state(game.players[player_id].export_policy_state())


def test_cached_observers_equal_full_event_replay_across_rounds():
    random.seed(99221)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [4, 3, 2]
    original_record = game._record_public_event
    checked = []

    def compare_at_each_public_event(event):
        original_record(event)
        if game._information_state.phase in ("dealt", "bidding", "playing"):
            for pid in (1, 2):
                first = game.observe(pid)
                second = game.observe(pid)  # No new events: cache hit.
                slow = _reference_observe(game, pid)
                assert first == second == slow
                checked.append((game.roundnumber, event["type"], pid))

    game._record_public_event = compare_at_each_public_event
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber == 3
    assert len(checked) > 50
    assert len({r for r, _, _ in checked}) == 3
    assert sum(typ == "card" for _, typ, _ in checked) == 3 * (4 + 3 + 2) * 2


def test_cached_observer_refreshed_after_new_private_policy_state():
    random.seed(449)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [3]
    game.resetGame()
    game.randomDeal(3)
    game._record_public_event({
        "type": "trump", "player": game.dealer, "suit": "h",
    })
    game.getPlayerBids()
    first = game.observe(1)
    # The policy's private memory is attached afresh; it is not copied
    # into the cached public world model.
    game.players[1].zeroCounter += 1
    after = game.observe(1)
    assert first.own_hand == after.own_hand
    assert first.history_json == after.history_json
    assert first.policy_state_json != after.policy_state_json
