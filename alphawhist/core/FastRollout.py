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
from alphawhist.core.EndgamePolicy import pick_bid_aware_two_card

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
        """Checked public entry point; invalid moves leave state untouched."""
        bit = 1 << card
        if not self.hands[self.turn] & bit:
            raise ValueError("Card is not held by the acting player")
        if not legal_mask(self.hands[self.turn], self.pile) & bit:
            raise ValueError("Card does not follow suit")
        self._play_legal(card)

    def _play_legal(self, card):
        """Unchecked hot path: the caller must supply a currently legal card.

        Rollout continuations select exclusively from legal_mask(), avoiding
        a second set-membership and follow-suit test on every simulated card.
        The public play() method remains checked.
        """
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
    if policy == "bid_aware_2":
        return pick_bid_aware_two_card(state, legal, rng)
    if callable(policy):
        card = policy(state, legal, rng)
        if not isinstance(card, int) or not (legal & (1 << card)):
            raise ValueError("Custom policy returned an illegal card")
        return card
    raise ValueError("Unknown rollout policy")


def finish_round(state, rng, policy="random"):
    """Finish a branch by running a non-interactive policy.

    Dispatch the common two-card continuation policy once per rollout rather
    than once per simulated card.  Preserve the exact RNG draw sequence and
    scoring path so existing seeded teacher labels remain bit-identical.
    """
    hands, pile, play = state.hands, state.pile, state._play_legal
    if policy == "bid_aware_2":
        while pile or any(hands):
            mask = legal_mask(hands[state.turn], pile)
            if not mask:
                raise ValueError("No legal move in an unfinished round")
            play(pick_bid_aware_two_card(state, mask, rng))
    else:
        while pile or any(hands):
            mask = legal_mask(hands[state.turn], pile)
            if not mask:
                raise ValueError("No legal move in an unfinished round")
            play(_choose_card(state, mask, rng, policy))
    return state.round_scores()


@dataclass(frozen=True)
class Evaluation:
    actions: tuple[str, ...]
    mean_round_scores: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    rollouts: int
    additional_trick_counts: tuple[tuple[int, ...], ...] | None = None

    def best_action(self):
        return self.actions[max(range(len(self.actions)), key=self.values.__getitem__)]


def _batch_impl(args, collect_histograms=False):
    """Shared exact simulation path for score sums and optional trick counts."""
    snapshot, actions, start, stop, seed, policy = args
    action_ids = [CARD_ID[card] for card in actions]
    n = snapshot.num_players
    accum = [[0] * n for _ in actions]
    counts = ([[0] * 8 for _ in actions] if collect_histograms else None)
    observer = snapshot.observer_id
    previously_won = snapshot.tricks_won[observer]
    max_additional = snapshot.cards_per_player - snapshot.trick_number
    deal_rng = random.Random()
    continuation_rng = random.Random()
    for index in range(start, stop):
        deal_rng.seed(_mix(seed, 2 * index))
        # The same sampled hidden world evaluates EVERY candidate action.
        world = snapshot.sample_hands(deal_rng)
        base = SimState.from_snapshot(snapshot, world)
        policy_seed = _mix(seed, 2 * index + 1)
        for i, card in enumerate(action_ids):
            state = SimState(base.hands.copy(), base.pile.copy(), base.bids,
                             base.tricks_won.copy(), base.leader, base.turn, base.trump)
            state.play(card)
            continuation_rng.seed(policy_seed)
            score = finish_round(state, continuation_rng, policy)
            for j, value in enumerate(score):
                accum[i][j] += value
            if counts is not None:
                won = state.tricks_won[observer] - previously_won
                if not 0 <= won <= max_additional <= 7:
                    raise AssertionError("Invalid trick-count target")
                counts[i][won] += 1
    return (accum, counts) if collect_histograms else accum


def _batch(args):
    """Score-only hot path, backwards-compatible with existing tests."""
    return _batch_impl(args)


def _histogram_batch(args):
    return _batch_impl(args, collect_histograms=True)


def evaluate_actions(snapshot, actions=None, n_rollouts=DEFAULT_ROLLOUTS, seed=0,
                     workers=1, batch_size=256, objective="score",
                     policy="random", collect_trick_histograms=False):
    """Evaluate candidate plays under paired possible-world rollouts.

    Reuses hidden deals and policy seeds across candidates (common random
    numbers), supports independent process batches, and reproduces exactly
    for a fixed seed regardless of workers/batch_size.

    'score' maximizes our expected round points; 'margin' maximizes our
    expected points minus the strongest *opponent expected score*.
    'bid_aware_2' chooses between two legal cards using a cheap bid-aware
    stochastic heuristic; other hand sizes use uniform random play.
    Random remains the default (backwards-compatible).
    Set collect_trick_histograms=True to return per-action integer counts
    of 0..7 ADDITIONAL tricks won by the observer. This is the training
    target for DistributionalWhistMLP, not an opponent's trick count.
    This is not the mean of the per-rollout minimum score margin.
    """
    if snapshot.phase != "playing" or snapshot.turn != snapshot.observer_id:
        raise ValueError("Snapshot must be at the observer's card decision")
    if n_rollouts < 1 or workers < 1 or batch_size < 1:
        raise ValueError("Rollouts, workers and batch size must be positive")
    if objective not in ("score", "margin"):
        raise ValueError("Objective must be 'score' or 'margin'")
    if not (policy in ("random", "highest", "bid_aware_2")
            if isinstance(policy, str) else callable(policy)):
        raise ValueError("Unknown rollout policy")
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
    totals = [[0] * snapshot.num_players for _ in actions]
    counts = ([[0] * 8 for _ in actions] if collect_trick_histograms else None)
    kernel = _histogram_batch if collect_trick_histograms else _batch

    def accumulate(part):
        batch, histogram = part if counts is not None else (part, None)
        for i, row in enumerate(batch):
            for j, value in enumerate(row):
                totals[i][j] += value
        if histogram is not None:
            for i, row in enumerate(histogram):
                for k, count in enumerate(row):
                    counts[i][k] += count

    if workers == 1:
        for part in map(kernel, tasks):
            accumulate(part)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(kernel, tasks):
                accumulate(part)
    if counts is not None and any(sum(row) != n_rollouts for row in counts):
        raise AssertionError("Incomplete action histogram")

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
    return Evaluation(tuple(actions), means, values, n_rollouts,
                      None if counts is None else tuple(tuple(row) for row in counts))
