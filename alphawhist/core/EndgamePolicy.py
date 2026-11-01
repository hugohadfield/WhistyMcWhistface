"""Fast, information-safe heuristic for two-card Whist endgames.

Only reads the acting player's two cards, the public current trick, public
trick counts, bids and trump. It MUST NOT look into opponents' sampled hands.
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


def _threats(card, led_suit, trump, void_rate):
    """Effective number of unseen higher cards; approximate, not calibrated."""
    rank = card % 13
    suit = card // 13
    if suit == trump:
        return 12 - rank
    if suit != led_suit:
        return 52.0  # Off-suit nontrump can never win this trick.
    return 12 - rank + 13 * void_rate


def _lead_win_probability(card, trump, others, cards_per_opponent,
                          other_own_card=-1):
    """Cheap prior for an unplayed card leading a trick.

    Not conditioned on specific opponent hands, which are secret. One threat
    factor is applied per possible opposing card, with a constant 50-card
    denominator as a first-order unseen-deck approximation.
    """
    if others == 0:
        return 1.0
    suit = card // 13
    threat = _threats(card, suit, trump, _TWO_CARD_VOID_RATE
                      if cards_per_opponent == 2 else _LAST_CARD_VOID_RATE)
    # Do not count the second card in our own hand as an opposing threat.
    if other_own_card >= 0:
        other_suit = other_own_card // 13
        other_rank = other_own_card % 13
        if suit == trump and other_suit == trump and other_rank > card % 13:
            threat -= 1
        elif suit != trump:
            if other_suit == suit and other_rank > card % 13:
                threat -= 1
            elif other_suit == trump:
                threat -= _TWO_CARD_VOID_RATE
    single_survival = max(0.0, min(1.0, 1.0 - threat / 50.0))
    return single_survival ** (others * cards_per_opponent)


def _current_win_probability(state, card, other):
    """Chance candidate wins current trick, given public cards seen so far."""
    pile = state.pile
    lead = pile[0] // 13 if pile else card // 13
    trump = state.trump
    suit = card // 13
    if suit == trump:
        power = 100 + card % 13
    elif suit == lead:
        power = card % 13
    else:
        return 0.0

    # If an already-played card beats ours, later plays cannot reverse that.
    for seen in pile:
        seen_suit = seen // 13
        seen_power = (100 + seen % 13 if seen_suit == trump else
                      seen % 13 if seen_suit == lead else -1)
        if seen_power > power:
            return 0.0
    remaining_players = len(state.hands) - len(pile) - 1
    if remaining_players == 0:
        return 1.0
    threat = _threats(card, lead, trump, _TWO_CARD_VOID_RATE)
    other_suit = other // 13
    if suit == trump and other_suit == trump and other % 13 > card % 13:
        threat -= 1
    elif suit != trump:
        if other_suit == suit and other % 13 > card % 13:
            threat -= 1
        elif other_suit == trump:
            threat -= _TWO_CARD_VOID_RATE
    survival = max(0.0, min(1.0, 1.0 - threat / 50.0))
    return survival ** (2 * remaining_players)


def _last_follow_probability(card, trump, others):
    """Chance to win the final trick after another player leads it."""
    if others == 0:
        return 1.0
    lead_prob = _lead_win_probability(card, trump, others, 1)
    rank_fraction = (card % 13 + 1) / 13.0
    if card // 13 == trump:
        # A trump can ruff non-trump leads, but not necessarily higher trump.
        return lead_prob * (0.75 + 0.25 * rank_fraction)
    # Only one remaining card: unknown lead must match suit (~1/4) and
    # the card must still outrank the leading card.
    return lead_prob * 0.25 * rank_fraction


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
    pa = _current_win_probability(state, a, b)
    pb = _current_win_probability(state, b, a)
    qa_lead = _lead_win_probability(b, state.trump, others, 1)
    qb_lead = _lead_win_probability(a, state.trump, others, 1)
    qa_follow = _last_follow_probability(b, state.trump, others)
    qb_follow = _last_follow_probability(a, state.trump, others)
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
