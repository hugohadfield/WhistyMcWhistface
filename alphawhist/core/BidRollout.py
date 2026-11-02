"""Monte Carlo bid evaluation using cheap bid_aware_2 card continuations.

Every legal bid is evaluated on the *same* sampled hidden deals and the same
random continuation seeds. Real opponents' cards are never observed. When we
bid before another player, their not-yet-announced bids are approximated by
the simple average-bid heuristic, respecting the dealer no-exact-sum and the
two-consecutive-zero constraints.

This is a one-level approximate model of bid-conditioned future play, NOT
nested Monte Carlo and NOT an exact simulation of our later MC(1024) player.
"""
from dataclasses import dataclass
import json
import random

from alphawhist.core.FastRollout import (
    SUITS, SimState, finish_round, hand_mask, _mix,
)
from alphawhist.core.InformationState import InformationState


@dataclass(frozen=True)
class BidEvaluation:
    bids: tuple[int, ...]
    expected_round_scores: tuple[float, ...]
    exact_bid_probabilities: tuple[float, ...]
    mean_tricks: tuple[float, ...]
    rollouts: int

    def best_bid(self):
        return self.bids[max(range(len(self.bids)),
                             key=self.expected_round_scores.__getitem__)]


def public_zero_streaks(snapshot):
    """Consecutive zero counts from PUBLIC bidding history only."""
    streaks = [0] * snapshot.num_players
    for raw in snapshot.history_json:
        event = json.loads(raw)
        if event["type"] == "bid":
            p = event["player"]
            streaks[p] = streaks[p] + 1 if event["bid"] == 0 else 0
    return streaks


def legal_future_bids(player, bids, streaks, cards, dealer):
    """Legal next bids given previously announced or hypothetical bids."""
    n = len(bids)
    last_bidder = player == dealer
    return tuple(
        bid for bid in range(cards + 1)
        if (bid != 0 or streaks[player] < 2
            or (last_bidder and cards == 1
                and sum(b for b in bids if b is not None) == 0))
        and (not last_bidder or sum(b for b in bids if b is not None) + bid != cards)
    )


def _approx_future_bids(snapshot, candidate, previous_streaks):
    """Fill unknown *future* bids with the existing average-bid heuristic.

    This is intentionally cheaper than calling the MC bidding model at each
    simulated opponent. It does not read any actual or sampled opponent hand.
    """
    bids = list(snapshot.bids)
    own = snapshot.observer_id
    if bids[own] is not None or snapshot.turn != own:
        raise ValueError("Bidding snapshot must be before observer's bid")
    bids[own] = candidate
    streaks = list(previous_streaks)
    streaks[own] = streaks[own] + 1 if candidate == 0 else 0
    # A greedy future prediction can dead-end the final bidder (notably
    # in 1-card rounds when they cannot bid zero). Backtrack ONLY over
    # hypothetical future bids, preserving actual announced bids.
    future = []
    player = (own + 1) % snapshot.num_players
    first_bidder = (snapshot.dealer + 1) % snapshot.num_players
    while player != first_bidder:
        if bids[player] is not None:
            raise AssertionError("Already-announced bid follows current bidder")
        future.append(player)
        player = (player + 1) % snapshot.num_players

    def complete(position):
        if position == len(future):
            return True
        player = future[position]
        options = legal_future_bids(
            player, bids, streaks, snapshot.cards_per_player, snapshot.dealer)
        natural = round(snapshot.cards_per_player / snapshot.num_players)
        # Original RandomPlayer.makeAverageBid's cyclic preference order.
        preference = sorted(
            options,
            key=lambda b: ((b - natural) % (snapshot.cards_per_player + 1)))
        original_streak = streaks[player]
        for bid in preference:
            bids[player] = bid
            streaks[player] = original_streak + 1 if bid == 0 else 0
            if complete(position + 1):
                return True
        bids[player] = None
        streaks[player] = original_streak
        return False

    if not complete(0):
        # The current bid may be individually legal but make it impossible
        # for future bidders to complete the round legally under public
        # zero-bid restrictions. Such a bid cannot be safely simulated.
        return None
    if any(b is None for b in bids):
        raise AssertionError("Incomplete modelled bids")
    return tuple(bids)

def evaluate_bids(snapshot, legal_bids, *, n_rollouts=1024, seed=0):
    """Score hypothetical bids by simulating complete rounds.

    The sampler is conditioned on the actor's hand and public play/bids, not
    on hidden cards or opponents' unannounced bids. Pre-announced bids remain
    fixed. Common hidden worlds and random seeds reduce comparison variance.
    """
    if not isinstance(snapshot, InformationState):
        raise TypeError("Expected an actor-specific InformationState")
    if snapshot.phase != "bidding" or snapshot.turn != snapshot.observer_id:
        raise ValueError("Expected snapshot at the observer's bidding turn")
    if snapshot.trump not in SUITS or snapshot.pile or snapshot.played_cards:
        raise ValueError("Expected a pre-play bidding snapshot with trumps")
    if not isinstance(n_rollouts, int) or n_rollouts < 1:
        raise ValueError("n_rollouts must be a positive integer")
    options = tuple(legal_bids)
    if not options or len(options) != len(set(options)):
        raise ValueError("Legal bids must be distinct and nonempty")
    streaks = public_zero_streaks(snapshot)
    valid = legal_future_bids(
        snapshot.observer_id, snapshot.bids, streaks, snapshot.cards_per_player,
        snapshot.dealer)
    if any(b not in valid or type(b) is not int for b in options):
        raise ValueError("Illegal hypothetical bid")

    modeled_bids = tuple(_approx_future_bids(snapshot, b, streaks)
                         for b in options)
    if all(b is None for b in modeled_bids):
        raise ValueError("No hypothetical bid permits a legal future bidding round")
    sums = [0] * len(options)
    made = [0] * len(options)
    tricks = [0] * len(options)
    me = snapshot.observer_id
    rng = random.Random()
    continuation_rng = random.Random()
    for index in range(n_rollouts):
        rng.seed(_mix(seed, 2 * index))
        world = snapshot.sample_hands(rng)
        masks = [hand_mask(hand) for hand in world]
        policy_seed = _mix(seed, 2 * index + 1)
        for j, bids in enumerate(modeled_bids):
            if bids is None:
                continue
            state = SimState(
                hands=masks.copy(), pile=[], bids=bids,
                tricks_won=[0] * snapshot.num_players,
                leader=snapshot.leader, turn=snapshot.leader,
                trump=SUITS.index(snapshot.trump),
            )
            continuation_rng.seed(policy_seed)
            scores = finish_round(state, continuation_rng, policy="bid_aware_2")
            sums[j] += scores[me]
            tricks[j] += state.tricks_won[me]
            made[j] += state.tricks_won[me] == bids[me]
    return BidEvaluation(
        bids=options,
        expected_round_scores=tuple(
            s / n_rollouts if bids is not None else float("-inf")
            for s, bids in zip(sums, modeled_bids)),
        exact_bid_probabilities=tuple(m / n_rollouts for m in made),
        mean_tricks=tuple(t / n_rollouts for t in tricks),
        rollouts=n_rollouts,
    )
