"""Fast, history-free Whist rollouts from an observer InformationState.

The authoritative game records history and checks knowledge constraints. This
module only simulates complete *possible* worlds for Monte Carlo evaluation.
It cannot read opponents' true hands and never mutates the input snapshot.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import random
from typing import Callable

from alphawhist.core.WhistLib import all_cards_in_deck

CARDS = tuple(all_cards_in_deck)
CARD_ID = {card: i for i, card in enumerate(CARDS)}
SUITS = "hcsd"
SUIT_MASKS = tuple(sum(1 << c for c in range(s * 13, (s + 1) * 13))
                   for s in range(4))
UINT64 = (1 << 64) - 1
DEFAULT_ROLLOUTS = 1024


def hand_mask(cards):
    """Pack a hand into one 52-bit Python integer."""
    value = 0
    for card in cards:
        bit = 1 << CARD_ID[card]
        if value & bit:
            raise ValueError("Hand contains a duplicate card")
        value |= bit
    return value


def card_ids(mask):
    """Yield the set bits as card indices without building a card list."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def legal_mask(hand, pile):
    if not hand:
        return 0
    if pile:
        following = hand & SUIT_MASKS[pile[0] // 13]
        if following:
            return following
    return hand


def trick_winner(pile, trump):
    """Index into the pile, with exactly the same ranking as analyse_pile."""
    if not pile:
        raise ValueError("Cannot score an empty trick")
    lead_suit = pile[0] // 13
    winner = 0
    best = -1
    for i, card in enumerate(pile):
        suit = card // 13
        if suit == trump:
            value = 100 + card % 13
        elif suit == lead_suit:
            value = card % 13
        else:
            continue
        if value > best:
            best, winner = value, i
    return winner


@dataclass(slots=True)
class SimState:
    """One cheap mutable rollout branch; never shared between candidates."""
    hands: list[int]
    pile: list[int]
    bids: tuple[int, ...]
    tricks_won: list[int]
    leader: int
    turn: int
    trump: int

    @classmethod
    def from_snapshot(cls, snapshot, dealt_hands):
        if snapshot.phase != "playing":
            raise ValueError("Rollouts require a playing-phase snapshot")
        if any(b is None for b in snapshot.bids):
            raise ValueError("All players must have bid")
        if len(dealt_hands) != snapshot.num_players:
            raise ValueError("Wrong number of hands")
        return cls(
            hands=[hand_mask(hand) for hand in dealt_hands],
            pile=[CARD_ID[card] for _, card in snapshot.pile],
            bids=tuple(snapshot.bids),
            tricks_won=list(snapshot.tricks_won),
            leader=snapshot.leader,
            turn=snapshot.turn,
            trump=SUITS.index(snapshot.trump),
        )

    def play(self, card):
        """Play one legal card and score/advance the trick if completed."""
        if not self.hands[self.turn] & (1 << card):
            raise ValueError("Card is not held by the acting player")
        if not legal_mask(self.hands[self.turn], self.pile) & (1 << card):
            raise ValueError("Card does not follow suit")
        self.hands[self.turn] ^= 1 << card
        self.pile.append(card)
        if len(self.pile) == len(self.hands):
            winner = (self.leader + trick_winner(self.pile, self.trump)) % len(self.hands)
            self.tricks_won[winner] += 1
            self.leader = self.turn = winner
            self.pile.clear()
        else:
            self.turn = (self.turn + 1) % len(self.hands)

    def round_scores(self):
        if self.pile or any(self.hands):
            raise ValueError("Round is not finished")
        return tuple(won + (10 if won == bid else 0)
                     for won, bid in zip(self.tricks_won, self.bids))


def _mix(seed, index):
    """Stable per-rollout seeds independent of process count and batching."""
    x = (int(seed) + 0x9E3779B97F4A7C15 * (index + 1)) & UINT64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & UINT64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & UINT64
    return x ^ (x >> 31)


def _choose_card(state, legal, rng, policy):
    if policy == "random":
        # Exactly uniform over legal cards; no allocation of a full list.
        bit_position = rng.randrange(legal.bit_count())
        while bit_position:
            legal &= legal - 1
            bit_position -= 1
        return (legal & -legal).bit_length() - 1
    if policy == "highest":
        return legal.bit_length() - 1
    if callable(policy):
        card = policy(state, legal, rng)
        if not isinstance(card, int) or not (legal & (1 << card)):
            raise ValueError("Custom policy returned an illegal card")
        return card
    raise ValueError("Unknown rollout policy")


def finish_round(state, rng, policy="random"):
    """Finish a branch by running a non-interactive policy."""
    while state.pile or any(state.hands):
        mask = legal_mask(state.hands[state.turn], state.pile)
        if not mask:
            raise ValueError("No legal move in an unfinished round")
        state.play(_choose_card(state, mask, rng, policy))
    return state.round_scores()


@dataclass(frozen=True)
class Evaluation:
    actions: tuple[str, ...]
    mean_round_scores: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    rollouts: int

    def best_action(self):
        return self.actions[max(range(len(self.actions)), key=self.values.__getitem__)]


def _batch(args):
    snapshot, actions, start, stop, seed, policy = args
    action_ids = [CARD_ID[card] for card in actions]
    n = snapshot.num_players
    accum = [[0] * n for _ in actions]
    for index in range(start, stop):
        # One sampled possible world, reused by EVERY candidate action.
        world = snapshot.sample_hands(random.Random(_mix(seed, 2 * index)))
        policy_seed = _mix(seed, 2 * index + 1)
        for i, card in enumerate(action_ids):
            state = SimState.from_snapshot(snapshot, world)
            state.play(card)
            score = finish_round(state, random.Random(policy_seed), policy)
            for j, value in enumerate(score):
                accum[i][j] += value
    return accum


def evaluate_actions(snapshot, actions=None, n_rollouts=DEFAULT_ROLLOUTS, seed=0,
                     workers=1, batch_size=256, objective="score",
                     policy="random"):
    """Evaluate candidate plays under paired possible-world rollouts.

    Reuses hidden deals and policy seeds across candidates (common random
    numbers), supports independent process batches, and reproduces exactly
    for a fixed seed regardless of workers/batch_size.

    'score' maximizes our expected round points; 'margin' maximizes our
    expected points minus the strongest opponent's points *per rollout*.
    """
    if snapshot.phase != "playing" or snapshot.turn != snapshot.observer_id:
        raise ValueError("Snapshot must be at the observer's card decision")
    if n_rollouts < 1 or workers < 1 or batch_size < 1:
        raise ValueError("Rollouts, workers and batch size must be positive")
    if objective not in ("score", "margin"):
        raise ValueError("Objective must be 'score' or 'margin'")
    own_id = snapshot.observer_id
    pile = [CARD_ID[card] for _, card in snapshot.pile]
    allowed = legal_mask(hand_mask(snapshot.own_hand), pile)
    if actions is None:
        actions = tuple(CARDS[i] for i in card_ids(allowed))
    else:
        actions = tuple(actions)
    if not actions or len(actions) != len(set(actions)):
        raise ValueError("Actions must be distinct and nonempty")
    if any(CARD_ID.get(action, -1) < 0 or
           not allowed & (1 << CARD_ID[action]) for action in actions):
        raise ValueError("Candidate actions must be legal observed cards")

    tasks = [(snapshot, actions, start, min(start + batch_size, n_rollouts),
              seed, policy)
             for start in range(0, n_rollouts, batch_size)]
    if workers == 1:
        pieces = map(_batch, tasks)
        totals = [[0] * snapshot.num_players for _ in actions]
        for batch in pieces:
            for i, row in enumerate(batch):
                for j, value in enumerate(row):
                    totals[i][j] += value
    else:
        totals = [[0] * snapshot.num_players for _ in actions]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(_batch, tasks):
                for i, row in enumerate(batch):
                    for j, value in enumerate(row):
                        totals[i][j] += value

    means = tuple(tuple(value / n_rollouts for value in row) for row in totals)
    if objective == "score":
        values = tuple(row[own_id] for row in means)
    else:
        # The expected margin is NOT generally E[score] - max(E[opponents]).
        # Evaluate the per-rollout margin as a separate metric if needed;
        # here use 'score minus strongest expected opponent', matching the
        # original SnakeMonteCarloPlayer's ranking.
        values = tuple(row[own_id] - max(row[j] for j in range(snapshot.num_players)
                                          if j != own_id) for row in means)
    return Evaluation(tuple(actions), means, values, n_rollouts)
