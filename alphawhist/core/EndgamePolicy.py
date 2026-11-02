"""Fast, information-safe heuristic for two- and three-card Whist endgames.

Only reads the acting player's hand, the public current trick, public
trick counts, bids and trump. Probability tables are computed at import. It MUST NOT look into opponents' sampled hands.
No additional Monte Carlo sampling, search trees or deck allocations.

All card IDs use FastRollout's original 52-card suit-major encoding:
13 ranks per suit, ranked 2..ace, suits h/c/s/d.
"""
import math

# Soft, deliberately approximate frequencies for non-trump leads.
# With one remaining card a follower has no ability to choose its suit.
_TWO_CARD_VOID_RATE = 0.56
_LAST_CARD_VOID_RATE = 0.75
_TEMPERATURE = 1.5


def _win_tables(cards_per_opponent, void_fraction):
    """Lookup probability of surviving N future opponents.

    Calculated once at module import; there are at most six opponents.
    Suit flag index 0 = ordinary suit, 1 = trump. Rank index 0..12.
    """
    return tuple(
        (
            tuple(
                (max(0.0, 1.0 - (12 - rank + 13 * void_fraction) / 50.0)
                 ** (opponents * cards_per_opponent))
                for rank in range(13)
            ),
            tuple(
                ((1.0 - (12 - rank) / 50.0)
                 ** (opponents * cards_per_opponent))
                for rank in range(13)
            )
        )
        for opponents in range(7)
    )


# The costly exponentials never happen inside the hot rollout loop.
_CURRENT_WIN = _win_tables(2, _TWO_CARD_VOID_RATE)
_LAST_LEAD_WIN = _win_tables(1, _LAST_CARD_VOID_RATE)
_LAST_FOLLOW_WIN = tuple(
    (
        tuple(_LAST_LEAD_WIN[n][0][rank] * 0.25 * ((rank + 1) / 13.0)
              for rank in range(13)),
        tuple(_LAST_LEAD_WIN[n][1][rank] *
              (0.75 + 0.25 * ((rank + 1) / 13.0))
              for rank in range(13)),
    )
    for n in range(7)
)


def _current_win_probability(state, card, table=None):
    """Survival chance given the public pile; never read hidden hands."""
    pile = state.pile
    trump = state.trump
    suit = card // 13
    lead = pile[0] // 13 if pile else suit
    rank = card % 13
    if suit == trump:
        power = 100 + rank
    elif suit == lead:
        power = rank
    else:
        return 0.0
    # Only public cards already on the table can rule out a win.
    for seen in pile:
        other_suit = seen // 13
        strength = (100 + seen % 13 if other_suit == trump else
                    seen % 13 if other_suit == lead else -1)
        if strength > power:
            return 0.0
    opponents_after = len(state.hands) - len(pile) - 1
    return (table if table is not None else _CURRENT_WIN)[opponents_after][int(suit == trump)][rank]


def _two_trick_value(p, q_lead, q_follow, needed):
    """E[additional tricks] + 10*P(exactly the missing number of tricks).

    p is the chance of winning now. q_lead and q_follow estimate the final
    trick win chance after a win/loss. Values are heuristic, not true PMFs.
    """
    p2 = p * q_lead
    p0 = (1.0 - p) * (1.0 - q_follow)
    p1 = 1.0 - p0 - p2
    bonus = 10.0 * (p0 if needed == 0 else
                    p1 if needed == 1 else
                    p2 if needed == 2 else 0.0)
    return p1 + 2.0 * p2 + bonus


def two_card_values(state, legal):
    """Return (card_A, card_B, Q_A, Q_B), or None if not two legal moves.

    Runtime is constant in the number of possible deals. The only reference
    to state.hands is the acting player's own hand and the player count.
    """
    player = state.turn
    hand = state.hands[player]
    if hand.bit_count() != 2 or legal.bit_count() != 2:
        return None
    first_bit = legal & -legal
    a = first_bit.bit_length() - 1
    b = (legal ^ first_bit).bit_length() - 1
    needed = state.bids[player] - state.tricks_won[player]
    others = len(state.hands) - 1
    pa = _current_win_probability(state, a)
    pb = _current_win_probability(state, b)
    # All terminal lead/follow probabilities are table lookups.
    sa, ra = int(a // 13 == state.trump), a % 13
    sb, rb = int(b // 13 == state.trump), b % 13
    qa_lead = _LAST_LEAD_WIN[others][sb][rb]
    qb_lead = _LAST_LEAD_WIN[others][sa][ra]
    qa_follow = _LAST_FOLLOW_WIN[others][sb][rb]
    qb_follow = _LAST_FOLLOW_WIN[others][sa][ra]
    return (a, b,
            _two_trick_value(pa, qa_lead, qa_follow, needed),
            _two_trick_value(pb, qb_lead, qb_follow, needed))


def pick_bid_aware_two_card(state, legal, rng):
    """Softmax over modeled expected round scores; uniform fallback.

    If a follow-suit restriction forces one card, or the acting hand has
    more/less than two cards, avoid heuristic computation. The last trick
    is always forced for this policy.
    """
    values = two_card_values(state, legal)
    if values is not None:
        a, b, va, vb = values
        # Stable logistic; utility differences are in round-point units.
        probability_a = 1.0 / (1.0 + math.exp((vb - va) / _TEMPERATURE))
        return a if rng.random() < probability_a else b
    # Same uniform-bit selection as the baseline, without allocating lists.
    k = rng.randrange(legal.bit_count())
    while k:
        legal &= legal - 1
        k -= 1
    return (legal & -legal).bit_length() - 1


# Three-card lookahead: approximate the remaining two-card policy by taking
# its *softmax-weighted expected value*, not by assuming it always picks the
# better of the two cards. No search, sampling, or opponents' actual cards.
_THREE_CARD_WIN = _win_tables(3, 0.46)

# If we lose the current trick, somebody else leads the next trick. We do
# not know that future lead; approximate following rather than pretending
# our own next card is a guaranteed lead.
_TWO_CARD_FOLLOW_WIN = tuple(
    (
        tuple(_CURRENT_WIN[n][0][rank] * 0.25 * ((rank + 1) / 13.0)
              for rank in range(13)),
        tuple(_CURRENT_WIN[n][1][rank] *
              (0.75 + 0.25 * ((rank + 1) / 13.0))
              for rank in range(13)),
    )
    for n in range(7)
)


def _softmax_probability_first(a, b):
    """Probability of choosing the first of two cards; numerically stable."""
    return 1.0 / (1.0 + math.exp((b - a) / _TEMPERATURE))


def _two_card_continuation_value(a, b, needed, leading, trump, others):
    """Expected future 2-card return under the *stochastic* 2-card heuristic.

    Approximate unknown future public tricks, retaining the distinction
    between leading the next trick (after a win) and following another
    player's lead (after a loss).
    """
    fa, ra = int(a // 13 == trump), a % 13
    fb, rb = int(b // 13 == trump), b % 13
    first_table = _CURRENT_WIN if leading else _TWO_CARD_FOLLOW_WIN
    pa = first_table[others][fa][ra]
    pb = first_table[others][fb][rb]
    va = _two_trick_value(
        pa, _LAST_LEAD_WIN[others][fb][rb],
        _LAST_FOLLOW_WIN[others][fb][rb], needed)
    vb = _two_trick_value(
        pb, _LAST_LEAD_WIN[others][fa][ra],
        _LAST_FOLLOW_WIN[others][fa][ra], needed)
    choose_a = _softmax_probability_first(va, vb)
    return choose_a * va + (1.0 - choose_a) * vb


def three_card_values(state, legal):
    """Return (candidate_cards, expected_round_values) for 3-card choices.

    At most three candidates. Each branches on winning the current trick,
    then evaluates the two-card continuation under its stochastic policy.
    The public current pile is used for the first-trick probability only.
    All remaining opponent cards are deliberately ignored.
    """
    me = state.turn
    if state.hands[me].bit_count() != 3 or legal.bit_count() < 2:
        return None

    cards = tuple(
        bit.bit_length() - 1
        for bit in (1 << i for i in range(52))
        if legal & bit
    )
    if len(cards) > 3:
        raise ValueError("Illegal three-card action mask")

    needed = state.bids[me] - state.tricks_won[me]
    others = len(state.hands) - 1
    hand = state.hands[me]
    values = []
    for card in cards:
        remaining = hand ^ (1 << card)
        first_bit = remaining & -remaining
        a = first_bit.bit_length() - 1
        b = (remaining ^ first_bit).bit_length() - 1
        chance = _current_win_probability(state, card, _THREE_CARD_WIN)
        on_win = _two_card_continuation_value(
            a, b, needed - 1, True, state.trump, others)
        on_loss = _two_card_continuation_value(
            a, b, needed, False, state.trump, others)
        values.append(chance * (1.0 + on_win) + (1.0-chance) * on_loss)
    return cards, tuple(values)


def pick_bid_aware_three_card(state, legal, rng):
    """Use 3-card lookahead, unchanged 2-card policy, else uniform random.

    The exact same callable is used by real players and by FastRollout.
    """
    result = three_card_values(state, legal)
    if result is None:
        return pick_bid_aware_two_card(state, legal, rng)
    cards, values = result
    highest = max(values)
    weights = tuple(math.exp((v - highest) / _TEMPERATURE) for v in values)
    cutoff = rng.random() * sum(weights)
    cumulative = 0.0
    for card, weight in zip(cards, weights):
        cumulative += weight
        if cutoff < cumulative:
            return card
    return cards[-1]
