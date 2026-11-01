"""Observer-specific, immutable, versioned Whist information snapshots.

Only the observing player's cards and public observations belong here.
In particular, never populate opponents' candidate sets from their real hands.
"""
from dataclasses import dataclass, replace
import json
import random

from alphawhist.core.WhistLib import all_cards_in_deck, analyse_pile

SCHEMA_VERSION = 1
SUITS = frozenset("hcsd")
DECK = frozenset(all_cards_in_deck)


def _event(event):
    # A canonical string makes history deeply immutable, while JSON remains
    # independent of Python object identities, classes and pickle.
    return json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_cards(cards):
    cards = tuple(cards)
    if len(cards) != len(set(cards)) or not set(cards) <= DECK:
        raise ValueError("Cards must be unique, valid deck cards")
    return cards


@dataclass(frozen=True)
class OpponentKnowledge:
    player_id: int
    remaining_count: int
    possible_cards: frozenset[str]
    void_suits: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.remaining_count < 0 or self.remaining_count > len(self.possible_cards):
            raise ValueError("Impossible opponent hand size")
        if not self.possible_cards <= DECK or not self.void_suits <= SUITS:
            raise ValueError("Invalid opponent card or suit")
        if any(card[-1] in self.void_suits for card in self.possible_cards):
            raise ValueError("Candidate cards contradict a known void suit")

    def to_dict(self):
        return {
            "player_id": self.player_id,
            "remaining_count": self.remaining_count,
            "possible_cards": sorted(self.possible_cards),
            "void_suits": sorted(self.void_suits),
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {"player_id", "remaining_count", "possible_cards", "void_suits"}:
            raise ValueError("Unexpected opponent knowledge schema")
        return cls(
            player_id=raw["player_id"],
            remaining_count=raw["remaining_count"],
            possible_cards=frozenset(_unique_cards(raw["possible_cards"])),
            void_suits=frozenset(raw["void_suits"]),
        )


@dataclass(frozen=True)
class InformationState:
    observer_id: int
    num_players: int
    round_index: int
    cards_per_player: int
    dealer: int
    leader: int
    turn: int
    trick_number: int
    phase: str
    trump: str | None
    own_hand: tuple[str, ...]
    opponents: tuple[OpponentKnowledge, ...]
    played_cards: tuple[str, ...]
    pile: tuple[tuple[int, str], ...]
    bids: tuple[int | None, ...]
    tricks_won: tuple[int, ...]
    scores: tuple[int, ...]
    history_json: tuple[str, ...] = ()
    policy_state_json: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported information state version")
        if self.num_players < 2 or self.cards_per_player < 1:
            raise ValueError("Invalid game size")
        if self.num_players * self.cards_per_player > 52:
            raise ValueError("Too many dealt cards")
        if not all(0 <= i < self.num_players for i in
                   (self.observer_id, self.dealer, self.leader, self.turn)):
            raise ValueError("Invalid player number")
        if self.phase not in {"dealt", "bidding", "playing", "scoring", "between_rounds"}:
            raise ValueError("Invalid phase")
        if self.trump is not None and self.trump not in SUITS:
            raise ValueError("Invalid trump")
        if not 0 <= self.trick_number <= self.cards_per_player:
            raise ValueError("Invalid trick index")
        if not (len(self.bids) == len(self.tricks_won) ==
                len(self.scores) == self.num_players):
            raise ValueError("Wrong number of player entries")
        if len(self.opponents) != self.num_players - 1 or (
            {p.player_id for p in self.opponents}
            != set(range(self.num_players)) - {self.observer_id}
        ):
            raise ValueError("Missing or duplicated opponent")
        own = set(_unique_cards(self.own_hand))
        played = set(_unique_cards(self.played_cards))
        if own & played:
            raise ValueError("Own hand overlaps played cards")
        if len(self.pile) > self.num_players:
            raise ValueError("Too many cards in pile")
        if len({pid for pid, _ in self.pile}) != len(self.pile):
            raise ValueError("Repeated player in trick")
        for pid, card in self.pile:
            if pid not in range(self.num_players) or card not in played:
                raise ValueError("Pile must contain publicly played cards")
        for p in self.opponents:
            if p.possible_cards & (own | played):
                raise ValueError("Opponent candidates contain known unavailable cards")
        if sum(p.remaining_count for p in self.opponents) + len(own) > 52 - len(played):
            raise ValueError("Insufficient unseen cards")
        for event in self.history_json:
            if not isinstance(json.loads(event), dict):
                raise ValueError("Invalid public event")
        if self.policy_state_json is not None:
            json.loads(self.policy_state_json)

    @classmethod
    def new_round(cls, own_hand, num_players, round_index, dealer,
                  leader, cards_per_player, scores=None, observer_id=0,
                  previous_history=(), policy_state=None):
        own_hand = _unique_cards(own_hand)
        if len(own_hand) != cards_per_player:
            raise ValueError("Observer must have the full dealt hand")
        unseen = DECK - set(own_hand)
        opponents = tuple(
            OpponentKnowledge(pid, cards_per_player, frozenset(unseen))
            for pid in range(num_players) if pid != observer_id
        )
        event = {"type": "round_start", "round": round_index,
                 "cards_per_player": cards_per_player, "dealer": dealer,
                 "leader": leader}
        return cls(
            observer_id=observer_id, num_players=num_players,
            round_index=round_index, cards_per_player=cards_per_player,
            dealer=dealer, leader=leader, turn=dealer, trick_number=0,
            phase="dealt", trump=None, own_hand=own_hand,
            opponents=opponents, played_cards=(), pile=(),
            bids=(None,) * num_players, tricks_won=(0,) * num_players,
            scores=tuple(scores if scores is not None else (0,) * num_players),
            history_json=tuple(previous_history) + (_event(event),),
            policy_state_json=None if policy_state is None else _event(policy_state),
        )

    @property
    def public_history(self):
        return tuple(json.loads(e) for e in self.history_json)

    @property
    def policy_state(self):
        return (None if self.policy_state_json is None
                else json.loads(self.policy_state_json))

    def with_policy_state(self, state):
        return replace(self, policy_state_json=None if state is None else _event(state))

    def apply_public_event(self, event):
        """Pure reducer for public events. Caller may never pass hidden cards."""
        kind = event.get("type")
        if kind == "trump":
            if self.phase != "dealt" or event.get("player") != self.dealer:
                raise ValueError("Unexpected trump choice")
            suit = event["suit"]
            if suit not in SUITS:
                raise ValueError("Invalid trump")
            result = replace(self, trump=suit, phase="bidding",
                             turn=(self.dealer + 1) % self.num_players)

        elif kind == "bid":
            if self.phase != "bidding" or event.get("player") != self.turn:
                raise ValueError("Bid is out of turn")
            bid = event["bid"]
            if type(bid) is not int or not 0 <= bid <= self.cards_per_player:
                raise ValueError("Invalid bid")
            bids = list(self.bids)
            if bids[self.turn] is not None:
                raise ValueError("Player has already bid")
            bids[self.turn] = bid
            all_bid = all(b is not None for b in bids)
            result = replace(self, bids=tuple(bids),
                             phase="playing" if all_bid else "bidding",
                             turn=self.leader if all_bid else (self.turn + 1) % self.num_players)

        elif kind == "card":
            if self.phase != "playing" or event.get("player") != self.turn:
                raise ValueError("Card played out of turn")
            card = event["card"]
            if card not in DECK or card in self.played_cards:
                raise ValueError("Invalid or previously played card")
            if len(self.pile) >= self.num_players:
                raise ValueError("Trick already complete")
            leading_suit = self.pile[0][1][-1] if self.pile else None
            player = self.turn
            own = self.own_hand
            if player == self.observer_id:
                if card not in own:
                    raise ValueError("Observer does not hold played card")
                if leading_suit and card[-1] != leading_suit and any(
                    c[-1] == leading_suit for c in own
                ):
                    raise ValueError("Observer must follow suit")
                own = tuple(c for c in own if c != card)
            elif card in self.own_hand:
                raise ValueError("Opponent cannot play the observer's card")

            updated_opponents = []
            for opponent in self.opponents:
                possible = set(opponent.possible_cards)
                possible.discard(card)
                void = set(opponent.void_suits)
                count = opponent.remaining_count
                if opponent.player_id == player:
                    if card not in opponent.possible_cards:
                        raise ValueError("Opponent played an impossible card")
                    count -= 1
                    if leading_suit and card[-1] != leading_suit:
                        possible = {c for c in possible if c[-1] != leading_suit}
                        void.add(leading_suit)
                updated_opponents.append(OpponentKnowledge(
                    opponent.player_id, count, frozenset(possible), frozenset(void)
                ))
            result = replace(
                self, own_hand=own, opponents=tuple(updated_opponents),
                played_cards=self.played_cards + (card,),
                pile=self.pile + ((player, card),),
                turn=(player + 1) % self.num_players,
            )

        elif kind == "trick_end":
            if self.phase != "playing" or len(self.pile) != self.num_players:
                raise ValueError("Cannot close an unfinished trick")
            winner = self.pile[0][0] + analyse_pile(
                [c for _, c in self.pile], self.trump
            )
            winner %= self.num_players
            if event.get("winner") != winner:
                raise ValueError("Incorrect trick winner")
            won = list(self.tricks_won)
            won[winner] += 1
            next_trick = self.trick_number + 1
            result = replace(
                self, tricks_won=tuple(won), trick_number=next_trick,
                pile=(), leader=winner, turn=winner,
                phase="scoring" if next_trick == self.cards_per_player else "playing",
            )

        elif kind == "round_end":
            if self.phase != "scoring":
                raise ValueError("Round cannot end before all tricks")
            scores = tuple(event["scores"])
            if len(scores) != self.num_players:
                raise ValueError("Wrong number of scores")
            dealer = (self.dealer + 1) % self.num_players
            result = replace(self, scores=scores, round_index=self.round_index + 1,
                             dealer=dealer, leader=(dealer + 1) % self.num_players,
                             turn=(dealer + 1) % self.num_players,
                             trick_number=0, phase="between_rounds")
        else:
            raise ValueError("Unknown public event")
        return replace(result, history_json=result.history_json + (_event(event),))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "observer_id": self.observer_id,
            "num_players": self.num_players,
            "round_index": self.round_index,
            "cards_per_player": self.cards_per_player,
            "dealer": self.dealer, "leader": self.leader, "turn": self.turn,
            "trick_number": self.trick_number, "phase": self.phase,
            "trump": self.trump, "own_hand": list(self.own_hand),
            "opponents": [o.to_dict() for o in self.opponents],
            "played_cards": list(self.played_cards),
            "pile": [[pid, card] for pid, card in self.pile],
            "bids": list(self.bids), "tricks_won": list(self.tricks_won),
            "scores": list(self.scores),
            "public_history": list(self.public_history),
            "policy_state": self.policy_state,
        }

    @classmethod
    def from_dict(cls, data):
        expected = set(cls.new_round(['2h'], 2, 0, 0, 1, 1).to_dict())
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Unexpected information state schema")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported information state version")
        return cls(
            observer_id=data["observer_id"],
            num_players=data["num_players"], round_index=data["round_index"],
            cards_per_player=data["cards_per_player"], dealer=data["dealer"],
            leader=data["leader"], turn=data["turn"],
            trick_number=data["trick_number"], phase=data["phase"],
            trump=data["trump"], own_hand=_unique_cards(data["own_hand"]),
            opponents=tuple(OpponentKnowledge.from_dict(o) for o in data["opponents"]),
            played_cards=_unique_cards(data["played_cards"]),
            pile=tuple((pid, card) for pid, card in data["pile"]),
            bids=tuple(data["bids"]), tricks_won=tuple(data["tricks_won"]),
            scores=tuple(data["scores"]),
            history_json=tuple(_event(e) for e in data["public_history"]),
            policy_state_json=(None if data["policy_state"] is None
                               else _event(data["policy_state"])),
        )

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **dict({"sort_keys": True}, **kwargs))

    @classmethod
    def from_json(cls, data):
        return cls.from_dict(json.loads(data))

    def save(self, path):
        from pathlib import Path
        Path(path).write_text(self.to_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        from pathlib import Path
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def sample_hands(self, rng=None):
        """Assign consistent opponent hands without accessing any true hand.

        Randomized matching guarantees feasibility, NOT a uniform distribution
        over feasible deals. Sampler probabilities are intentionally not used
        as calibrated beliefs about opponents.
        """
        rng = rng if rng is not None else random.Random()
        slots = []
        for opponent in self.opponents:
            candidates = list(sorted(opponent.possible_cards))
            for _ in range(opponent.remaining_count):
                options = candidates.copy()
                rng.shuffle(options)
                slots.append((opponent.player_id, options))
        ordering = list(range(len(slots)))
        rng.shuffle(ordering)
        matching = {}

        def assign(slot, visited):
            for card in slots[slot][1]:
                if card in visited:
                    continue
                visited.add(card)
                if card not in matching or assign(matching[card], visited):
                    matching[card] = slot
                    return True
            return False

        for slot in ordering:
            if not assign(slot, set()):
                raise ValueError("No consistent hidden-hand assignment")
        hands = [[] for _ in range(self.num_players)]
        hands[self.observer_id] = list(self.own_hand)
        for card, slot in matching.items():
            hands[slots[slot][0]].append(card)
        return tuple(tuple(hand) for hand in hands)
