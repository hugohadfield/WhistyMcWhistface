"""Mutable, observer-only rollout knowledge, using 52-bit public card sets.

This deliberately DOES NOT accept the simulator's full dealt hands. Construct
only from an already validated InformationState; update from public card plays
and publicly announced trick winners. A separate equivalence test checks
exact feature-vector equality against the authoritative immutable reducer.

No Game or SimState enters this class or its neural encoder. Bitmasks store:
  * observer's own remaining cards
  * publicly played cards, pile, and cards played by each player this round
  * opponents' feasible card sets and known void suits
The expensive history/knowledge validation remains at the input boundary,
not after each speculative simulated card.
"""
from dataclasses import dataclass
import json

import numpy as np

from alphawhist.core.FastRollout import CARD_ID, SUIT_MASKS, card_ids
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.encoding import (
    CARDS, SUITS, CARD_SHAPE, PLAYER_SHAPE, GLOBAL_SHAPE, MAX_PLAYERS,
    EncodedDecision, EncodedBatch,
)


def _mask(cards):
    value = 0
    for card in cards:
        value |= 1 << CARD_ID[card]
    return value


@dataclass(slots=True)
class FastObserver:
    """Only an observer's own cards and events publicly revealed to everyone."""
    observer_id: int
    num_players: int
    round_index: int
    cards_per_player: int
    dealer: int
    leader: int
    turn: int
    trick_number: int
    trump: int
    own_hand: int
    played: int
    pile: list[tuple[int, int]]
    bids: tuple[int, ...]
    tricks_won: list[int]
    scores: tuple[int, ...]
    played_by: list[int]
    possible: list[int]
    void_suits: list[int]
    remaining_counts: list[int]
    phase: str = "playing"

    @classmethod
    def from_snapshot(cls, snapshot: InformationState):
        if not isinstance(snapshot, InformationState) or snapshot.phase != "playing":
            raise ValueError("Requires a validated, playing-phase InformationState")
        n = snapshot.num_players
        me = snapshot.observer_id
        played_by = [0] * n
        # The model encodes ONLY this round's played-by channels, even if
        # the public event log contains all previous rounds.
        for raw in reversed(snapshot.history_json):
            event = json.loads(raw)
            if event["type"] == "round_start":
                break
            if event["type"] == "card":
                played_by[event["player"]] |= 1 << CARD_ID[event["card"]]
        possible, void, remaining = [0] * n, [0] * n, [0] * n
        remaining[me] = len(snapshot.own_hand)
        for opp in snapshot.opponents:
            possible[opp.player_id] = _mask(opp.possible_cards)
            void[opp.player_id] = sum(1 << SUITS.index(s) for s in opp.void_suits)
            remaining[opp.player_id] = opp.remaining_count
        return cls(
            observer_id=me, num_players=n,
            round_index=snapshot.round_index,
            cards_per_player=snapshot.cards_per_player,
            dealer=snapshot.dealer, leader=snapshot.leader,
            turn=snapshot.turn, trick_number=snapshot.trick_number,
            trump=SUITS.index(snapshot.trump),
            own_hand=_mask(snapshot.own_hand),
            played=_mask(snapshot.played_cards),
            pile=[(p, CARD_ID[c]) for p, c in snapshot.pile],
            bids=tuple(snapshot.bids),
            tricks_won=list(snapshot.tricks_won),
            scores=tuple(snapshot.scores),
            played_by=played_by,
            possible=possible,
            void_suits=void,
            remaining_counts=remaining,
        )

    @classmethod
    def from_sampled_perspective(cls, snapshot: InformationState, observer_id: int,
                                 remaining_hand, round_events):
        """Reconstruct ONE simulated player's private perspective, not the world.

        `remaining_hand` contains only this player's own cards in the sampled
        world. Cards they have already played are public. Reassemble their
        original private hand, then replay ONLY this round's public actions.

        In particular, opponent possibility sets start as the entire deck
        excluding this observer's original cards; no other sampled hands or
        sampled identities are passed to this function.
        """
        if not isinstance(snapshot, InformationState) or snapshot.phase != "playing":
            raise ValueError("Expected a genuine playing-phase InformationState")
        n = snapshot.num_players
        if not 0 <= observer_id < n:
            raise ValueError("Unknown perspective")
        if not round_events or round_events[0].get("type") != "round_start":
            raise ValueError("Current round_start event required")
        start = round_events[0]
        if (start["round"] != snapshot.round_index or
                start["cards_per_player"] != snapshot.cards_per_player or
                start["dealer"] != snapshot.dealer):
            raise ValueError("Public round setup does not match snapshot")
        prior_own = 0
        for event in round_events[1:]:
            if event["type"] == "card" and event["player"] == observer_id:
                prior_own |= 1 << CARD_ID[event["card"]]
        current_own = _mask(remaining_hand)
        if current_own & _mask(snapshot.played_cards):
            raise ValueError("Sampled remaining hand overlaps public cards")
        original_own = current_own | prior_own
        if original_own.bit_count() != snapshot.cards_per_player:
            raise ValueError("Sampled own hand contradicts publicly played cards")
        deck = (1 << 52) - 1
        possible = [deck & ~original_own for _ in range(n)]
        possible[observer_id] = 0
        obs = cls(
            observer_id=observer_id, num_players=n,
            round_index=snapshot.round_index,
            cards_per_player=snapshot.cards_per_player,
            dealer=snapshot.dealer, leader=start["leader"],
            turn=start["leader"], trick_number=0,
            trump=SUITS.index(snapshot.trump),
            own_hand=original_own, played=0, pile=[],
            bids=tuple(snapshot.bids), tricks_won=[0] * n,
            scores=tuple(snapshot.scores),
            played_by=[0] * n, possible=possible,
            void_suits=[0] * n,
            remaining_counts=[snapshot.cards_per_player] * n,
        )
        for event in round_events[1:]:
            kind = event["type"]
            if kind == "card":
                obs.play_card(event["player"], CARD_ID[event["card"]])
            elif kind == "trick_end":
                obs.end_trick(event["winner"])
            elif kind not in ("trump", "bid"):
                raise ValueError("Unexpected current-round event")
        # These checks ensure that reconstruction matches the *public* game,
        # without comparing with (or exposing) the other sampled hands.
        if (obs.turn != snapshot.turn or obs.leader != snapshot.leader or
                obs.trick_number != snapshot.trick_number or
                obs.tricks_won != list(snapshot.tricks_won) or
                obs.played != _mask(snapshot.played_cards) or
                obs.pile != [(p, CARD_ID[c]) for p, c in snapshot.pile] or
                obs.own_hand != current_own):
            raise AssertionError("Reconstructed public perspective diverged")
        return obs

    def fork(self):
        """Independent action branch; never alias another candidate's knowledge."""
        return FastObserver(
            self.observer_id, self.num_players, self.round_index,
            self.cards_per_player, self.dealer, self.leader, self.turn,
            self.trick_number, self.trump, self.own_hand, self.played,
            self.pile.copy(), self.bids, self.tricks_won.copy(), self.scores,
            self.played_by.copy(), self.possible.copy(), self.void_suits.copy(),
            self.remaining_counts.copy(), self.phase,
        )

    def play_card(self, player: int, cid: int):
        """Update exclusively with a public (player, card) event."""
        if self.phase != "playing" or player != self.turn:
            raise ValueError("Card played out of turn")
        bit = 1 << cid
        if self.played & bit:
            raise ValueError("Previously played card")
        lead = self.pile[0][1] // 13 if self.pile else None
        if player == self.observer_id:
            if not self.own_hand & bit:
                raise ValueError("Observer does not own that card")
            if lead is not None and cid // 13 != lead and (
                self.own_hand & SUIT_MASKS[lead]
            ):
                raise ValueError("Observer must follow suit")
            self.own_hand ^= bit
            self.remaining_counts[player] -= 1
        else:
            if self.own_hand & bit or not self.possible[player] & bit:
                raise ValueError("Opponent played a card unavailable to them")
            self.remaining_counts[player] -= 1
            if lead is not None and cid // 13 != lead:
                self.possible[player] &= ~SUIT_MASKS[lead]
                self.void_suits[player] |= 1 << lead
        # Every publicly played card becomes unavailable to every opponent.
        for opp in range(self.num_players):
            if opp != self.observer_id:
                self.possible[opp] &= ~bit
        self.played_by[player] |= bit
        self.played |= bit
        self.pile.append((player, cid))
        self.turn = (player + 1) % self.num_players

    def end_trick(self, winner: int):
        """Update with the trick-winner event after the simulator scores it."""
        if self.phase != "playing" or len(self.pile) != self.num_players:
            raise ValueError("Cannot close incomplete trick")
        if not 0 <= winner < self.num_players:
            raise ValueError("Invalid winner")
        self.tricks_won[winner] += 1
        self.trick_number += 1
        self.pile.clear()
        self.leader = self.turn = winner
        if self.trick_number == self.cards_per_player:
            self.phase = "scoring"

    def encode_decision(self) -> EncodedDecision:
        """Exactly the established version-1 1530-float model feature schema."""
        if self.phase != "playing" or self.turn != self.observer_id:
            raise ValueError("Fast neural features require the observer's turn")
        n, me = self.num_players, self.observer_id
        remaining = self.cards_per_player - self.trick_number
        if remaining != self.own_hand.bit_count():
            raise ValueError("Fast observation has inconsistent hand size")
        lead = self.pile[0][1] // 13 if self.pile else None
        legal_mask_bits = self.own_hand
        if lead is not None and self.own_hand & SUIT_MASKS[lead]:
            legal_mask_bits = self.own_hand & SUIT_MASKS[lead]

        card = np.zeros(CARD_SHAPE, dtype=np.float32)
        player = np.zeros(PLAYER_SHAPE, dtype=np.float32)
        global_features = np.zeros(GLOBAL_SHAPE, dtype=np.float32)
        legal = np.zeros((52,), dtype=np.bool_)

        for cid in card_ids(self.own_hand):
            card[cid, 0] = 1.0
        for cid in card_ids(self.played):
            card[cid, 1] = 1.0
        for cid in card_ids(legal_mask_bits):
            card[cid, 3] = 1.0
            legal[cid] = True
        card[self.trump * 13:(self.trump + 1) * 13, 4] = 1.0
        if lead is not None:
            card[lead * 13:(lead + 1) * 13, 5] = 1.0

        for pid, cid in self.pile:
            rel = (pid - me) % n
            card[cid, 2] = 1.0
            card[cid, 6 + MAX_PLAYERS + rel] = 1.0
        for pid in range(n):
            rel = (pid - me) % n
            for cid in card_ids(self.played_by[pid]):
                card[cid, 6 + rel] = 1.0
            if pid != me:
                for cid in card_ids(self.possible[pid]):
                    card[cid, 6 + 2 * MAX_PLAYERS + rel] = 1.0

        pile_players = {pid for pid, _ in self.pile}
        my_score = self.scores[me]
        for rel in range(n):
            pid = (me + rel) % n
            bid = self.bids[pid]
            won = self.tricks_won[pid]
            player[rel, :11] = (
                1.0, float(rel == 0), float(pid == self.dealer),
                float(pid == self.leader), float(pid == self.turn),
                bid / 7.0, won / 7.0, (bid - won) / 7.0,
                self.remaining_counts[pid] / 7.0,
                self.scores[pid] / 250.0,
                (self.scores[pid] - my_score) / 250.0,
            )
            player[rel, 15] = float(pid in pile_players)
            if pid != me:
                for s in range(4):
                    player[rel, 11 + s] = float(bool(self.void_suits[pid] & (1 << s)))

        global_features[:6] = (
            n / 7.0, self.cards_per_player / 7.0,
            remaining / 7.0, self.trick_number / 7.0,
            self.round_index / 12.0, len(self.pile) / 7.0,
        )
        global_features[6 + self.trump] = 1.0
        if lead is not None:
            global_features[10 + lead] = 1.0

        features = np.concatenate((card.ravel(), player.ravel(),
                                   global_features))
        return EncodedDecision(features, legal, remaining,
                               self.tricks_won[me], self.bids[me])


def encode_fast_batch(observations) -> EncodedBatch:
    """Encode only observer-safe FastObserver states for batched inference."""
    decisions = [item.encode_decision() for item in observations]
    if not decisions:
        raise ValueError("Cannot encode empty FastObserver batch")
    return EncodedBatch(
        np.stack([item.features for item in decisions]),
        np.stack([item.legal_mask for item in decisions]),
        np.asarray([item.remaining_tricks for item in decisions], dtype=np.int64),
        np.asarray([item.tricks_won for item in decisions], dtype=np.int64),
        np.asarray([item.bid for item in decisions], dtype=np.int64),
    )
