"""Fixed-width, information-safe neural features for Whist card decisions.

Input: an observer-specific InformationState at *that observer's turn*.
NEVER accepts Game or FastRollout.SimState: both can expose opponents' real
hands. Opponent possibility channels encode feasible cards, not ownership.

Card IDs match FastRollout's suit-major 52-card encoding (h, c, s, d;
ranks 2..14). Player indices are relative to the acting observer, enabling
one feature schema for 2..7 players. Inactive player slots are zero-padded.

Feature schema is explicitly versioned because training datasets and model
checkpoints must not silently mix feature layouts.
"""
from dataclasses import dataclass
import json

import numpy as np

from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistLib import all_cards_in_deck

FEATURE_SCHEMA_VERSION = 1
MAX_PLAYERS = 7
MAX_ADDITIONAL_TRICKS = 7
MAX_CARDS = 52
SUITS = "hcsd"
CARDS = tuple(all_cards_in_deck)
CARD_TO_ID = {card: i for i, card in enumerate(CARDS)}

CARD_CHANNELS = (
    "own", "played", "in_pile", "legal", "is_trump_suit", "is_lead_suit",
) + tuple(f"played_by_relative_{p}" for p in range(MAX_PLAYERS)) + tuple(
    f"pile_by_relative_{p}" for p in range(MAX_PLAYERS)
) + tuple(f"possible_for_relative_{p}" for p in range(MAX_PLAYERS))
PLAYER_CHANNELS = (
    "active", "self", "dealer", "leader", "turn", "bid", "tricks_won",
    "tricks_needed", "cards_remaining", "score", "score_minus_self",
    "void_h", "void_c", "void_s", "void_d", "played_in_current_trick",
)
GLOBAL_CHANNELS = (
    "num_players", "cards_per_player", "remaining_tricks", "trick_index",
    "round_index", "pile_size",
    "trump_h", "trump_c", "trump_s", "trump_d",
    "lead_h", "lead_c", "lead_s", "lead_d",
)
CARD_SHAPE = (MAX_CARDS, len(CARD_CHANNELS))
PLAYER_SHAPE = (MAX_PLAYERS, len(PLAYER_CHANNELS))
GLOBAL_SHAPE = (len(GLOBAL_CHANNELS),)
FEATURE_DIM = int(np.prod(CARD_SHAPE) + np.prod(PLAYER_SHAPE) + np.prod(GLOBAL_SHAPE))


@dataclass(frozen=True)
class EncodedDecision:
    """Single actor-observation, ready for batching or PyTorch conversion.

    `features` are float32 [FEATURE_DIM]. `legal_mask` is bool [52].
    `remaining_tricks` counts the current trick (if any) and future ones.
    """

    features: np.ndarray
    legal_mask: np.ndarray
    remaining_tricks: int
    tricks_won: int
    bid: int

    def __post_init__(self):
        if self.features.shape != (FEATURE_DIM,) or self.features.dtype != np.float32:
            raise ValueError("Expected float32 flattened feature vector")
        if self.legal_mask.shape != (MAX_CARDS,) or self.legal_mask.dtype != np.bool_:
            raise ValueError("Expected 52-element boolean legal action mask")
        if not self.legal_mask.any():
            raise ValueError("At least one legal action is required")
        if not 1 <= self.remaining_tricks <= MAX_ADDITIONAL_TRICKS:
            raise ValueError("Invalid number of remaining tricks")
        if self.tricks_won < 0 or self.bid < 0:
            raise ValueError("Bid and previous wins must be nonnegative")


@dataclass(frozen=True)
class EncodedBatch:
    features: np.ndarray
    legal_mask: np.ndarray
    remaining_tricks: np.ndarray
    tricks_won: np.ndarray
    bids: np.ndarray


def encode_decision(state: InformationState) -> EncodedDecision:
    """Encode only information available to the actor at a card decision."""
    if not isinstance(state, InformationState):
        raise TypeError("Expected an observer-specific InformationState, not a Game/SimState")
    if state.phase != "playing" or state.turn != state.observer_id:
        raise ValueError("Observation must be at the acting observer's play decision")
    if state.trump not in SUITS or any(bid is None for bid in state.bids):
        raise ValueError("Playing decision needs a known trump and all bids")
    if state.num_players > MAX_PLAYERS or state.cards_per_player > MAX_ADDITIONAL_TRICKS:
        raise ValueError("Unsupported game size")

    n, me = state.num_players, state.observer_id
    remaining = state.cards_per_player - state.trick_number
    if remaining != len(state.own_hand):
        raise ValueError("Observer hand size contradicts trick count")
    lead = state.pile[0][1][-1] if state.pile else None
    playable = tuple(c for c in state.own_hand if c[-1] == lead) if lead else ()
    legal_cards = playable if playable else state.own_hand

    card = np.zeros(CARD_SHAPE, dtype=np.float32)
    player = np.zeros(PLAYER_SHAPE, dtype=np.float32)
    global_features = np.zeros(GLOBAL_SHAPE, dtype=np.float32)
    legal = np.zeros((MAX_CARDS,), dtype=np.bool_)

    for c in state.own_hand:
        card[CARD_TO_ID[c], 0] = 1.0
    for c in state.played_cards:
        card[CARD_TO_ID[c], 1] = 1.0
    for c in legal_cards:
        cid = CARD_TO_ID[c]
        legal[cid] = True
        card[cid, 3] = 1.0

    # Rank is implicit in the 52 fixed card positions. Trump/lead membership
    # are explicit because their roles vary from round to round.
    for cid, c in enumerate(CARDS):
        card[cid, 4] = float(c[-1] == state.trump)
        card[cid, 5] = float(lead is not None and c[-1] == lead)

    for pid, c in state.pile:
        rel = (pid - me) % n
        cid = CARD_TO_ID[c]
        card[cid, 2] = 1.0
        card[cid, 6 + MAX_PLAYERS + rel] = 1.0

    # Only events from the CURRENT round. Earlier histories must not
    # masquerade as cards played in this round.
    for raw_event in reversed(state.history_json):
        event = json.loads(raw_event)
        if event["type"] == "round_start":
            break
        if event["type"] == "card":
            rel = (event["player"] - me) % n
            card[CARD_TO_ID[event["card"]], 6 + rel] = 1.0

    opponents = {o.player_id: o for o in state.opponents}
    pile_players = {pid for pid, _ in state.pile}
    my_score = state.scores[me]

    for rel in range(n):
        pid = (me + rel) % n
        bid = state.bids[pid]
        won = state.tricks_won[pid]
        other = opponents.get(pid)
        hand_remaining = len(state.own_hand) if pid == me else other.remaining_count
        player[rel, :11] = (
            1.0, float(rel == 0), float(pid == state.dealer),
            float(pid == state.leader), float(pid == state.turn),
            bid / 7.0, won / 7.0, (bid - won) / 7.0,
            hand_remaining / 7.0, state.scores[pid] / 250.0,
            (state.scores[pid] - my_score) / 250.0,
        )
        player[rel, 15] = float(pid in pile_players)
        if other is not None:
            for i, suit in enumerate(SUITS):
                player[rel, 11 + i] = float(suit in other.void_suits)
            for c in other.possible_cards:
                card[CARD_TO_ID[c], 6 + 2 * MAX_PLAYERS + rel] = 1.0

    global_features[:6] = (
        n / 7.0, state.cards_per_player / 7.0, remaining / 7.0,
        state.trick_number / 7.0, state.round_index / 12.0,
        len(state.pile) / 7.0,
    )
    global_features[6 + SUITS.index(state.trump)] = 1.0
    if lead is not None:
        global_features[10 + SUITS.index(lead)] = 1.0

    features = np.concatenate((card.ravel(), player.ravel(), global_features))
    return EncodedDecision(features, legal, remaining,
                           state.tricks_won[me], state.bids[me])


def encode_batch(states) -> EncodedBatch:
    """Batch an iterable of observation snapshots; reject an empty batch."""
    encoded = [encode_decision(state) for state in states]
    if not encoded:
        raise ValueError("Cannot encode an empty batch")
    return EncodedBatch(
        np.stack([item.features for item in encoded]),
        np.stack([item.legal_mask for item in encoded]),
        np.asarray([item.remaining_tricks for item in encoded], dtype=np.int64),
        np.asarray([item.tricks_won for item in encoded], dtype=np.int64),
        np.asarray([item.bid for item in encoded], dtype=np.int64),
    )
