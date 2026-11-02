"""Deterministic, suit-aware 3-card evaluator with a joint 2-trick continuation.

Two improvements over EndgamePolicy.three_card_values:
* hypergeometric (without-replacement) opponent void/follow probabilities,
  using ONLY the observer's own hand, seen cards and legal candidate sets;
* an analytical joint distribution over the last two tricks. For each
  opponent, we enumerate possible unordered TWO-CARD hands with exact
  combinatorial weights and random legal first plays, then combine
  per-opponent survival probabilities. No random sampling or MC search.

The 2P two-trick continuation is exact under uniform feasible *opponent*
pairs and random card choice, including lead changes, trump and following
suit. With 3+ players the separate opponent pair marginals are combined
as an independence approximation; no claim of exact joint dealing.
The three-to-two-card transition also approximates the unknown opponent
cards played on the current trick: it updates the number of remaining
cards and the identity of the next leader, not a full posterior conditioned
on the observed winner. The original policies remain untouched.
"""
from functools import lru_cache
from itertools import combinations
from math import comb, exp
import random

from alphawhist.core.EndgamePolicy import _TEMPERATURE
from alphawhist.core.FastRollout import CARD_ID, CARDS, SUITS, hand_mask

# Suit-major encoding: 13 ranks per suit, 2 ... ace.
SUIT_MASKS = tuple(sum(1 << c for c in range(s*13, (s+1)*13))
                   for s in range(4))


def void_probability(population, suit_cards, hand_size):
    """Exact hypergeometric probability of holding zero target-suit cards."""
    if not 0 <= suit_cards <= population or not 0 <= hand_size <= population:
        raise ValueError("Invalid hypergeometric parameters")
    return (comb(population-suit_cards, hand_size)/comb(population,hand_size)
            if population-suit_cards >= hand_size else 0.0)


def strength(card, lead_suit, trump):
    suit = card // 13
    if suit == trump:
        return 100 + card % 13
    if suit == lead_suit:
        return card % 13
    return -1


def _beats(first, second, lead_suit, trump):
    """Return whether second beats first, given the trick's lead suit."""
    return strength(second, lead_suit, trump) > strength(first, lead_suit, trump)


def _opponent_beat_probability(pool, count, lead_suit, trump, candidate):
    """Marginal probability one RANDOM legal response overtakes candidate.

    Uniform m-card hands from the available pool. Conditional on a hand,
    the opponent uniformly selects among follow-suit cards if possible.
    For each card in the required suit, symmetry makes its unconditional
    selection probability (1-P(void))/S. Outside the led suit it is
    P(void)/(N-S). These are exact marginal formulas for this model.
    """
    n = len(pool)
    if count == 0:
        return 0.0
    if count > n:
        raise ValueError("Impossible opponent candidate pool")
    follow = tuple(c for c in pool if c//13 == lead_suit)
    s = len(follow)
    void = void_probability(n, s, count)
    threshold = strength(candidate, lead_suit, trump)
    follow_better = sum(strength(c,lead_suit,trump)>threshold for c in follow)
    discard_better = sum(
        strength(c,lead_suit,trump)>threshold for c in pool
        if c//13 != lead_suit)
    p_follow = (1-void)*follow_better/s if s else 0.0
    p_discard = void*discard_better/(n-s) if n>s else 0.0
    return min(1.0,max(0.0,p_follow+p_discard))


def current_trick_win_probability(snapshot, card):
    """Estimate P(our played card takes this trick), with public voids.

    Earlier played cards are known exactly; future opponents' *marginal*
    response chances are computed combinatorially and multiplied.
    Distinct opponents are approximately independent (not exact joint
    sampling without replacement).
    """
    me = snapshot.observer_id
    if snapshot.turn != me or card not in tuple(CARD_ID[x] for x in snapshot.own_hand):
        raise ValueError("Candidate must be an actor-owned card at their turn")
    pile = [CARD_ID[c] for _, c in snapshot.pile]
    trump = SUITS.index(snapshot.trump)
    lead = pile[0]//13 if pile else card//13
    my_value = strength(card, lead, trump)
    if my_value < 0 or any(strength(x,lead,trump)>my_value for x in pile):
        return 0.0
    probability = 1.0
    opp_by_id = {opp.player_id:opp for opp in snapshot.opponents}
    next_pid = (me+1)%snapshot.num_players
    for _ in range(snapshot.num_players-1-len(pile)):
        opp = opp_by_id[next_pid]
        pool = tuple(CARD_ID[c] for c in opp.possible_cards)
        probability *= 1-_opponent_beat_probability(
            pool, opp.remaining_count, lead, trump, card)
        next_pid = (next_pid+1)%snapshot.num_players
    return max(0.0,min(1.0,probability))


def _two_player_pair_outcomes(x, y, pool, trump, actor_leads):
    """P(win first), P(win second), P(win both), enumerated analytically.

    Each unordered hidden opponent pair is equally likely. A random
    opponent chooses uniformly among legal first cards (the final card
    is forced). If the opponent leads, our first preference x is ignored
    when we must follow with y instead. The winner leads trick two.
    """
    n = len(pool)
    if n < 2:
        raise ValueError("Need at least two feasible cards for a two-card hand")
    win_first=win_second=win_both=0.0
    for u,v in combinations(pool,2):
        if actor_leads:
            a,b = u//13==x//13, v//13==x//13
            legal = (u,v) if a==b else ((u,) if a else (v,))
        else:
            legal = (u,v)
        mass = 1.0/(len(legal)*comb(n,2))
        for opponent_first in legal:
            opponent_last = v if opponent_first==u else u
            first_ours = x
            last_ours = y
            if not actor_leads:
                # Opponent already led: follow suit is mandatory.
                followed = opponent_first//13
                if x//13!=followed and y//13==followed:
                    first_ours,last_ours=y,x
            if actor_leads:
                us_first=not _beats(first_ours,opponent_first,
                                    first_ours//13,trump)
            else:
                us_first=_beats(opponent_first,first_ours,
                               opponent_first//13,trump)
            if us_first:
                us_second=not _beats(last_ours,opponent_last,
                                     last_ours//13,trump)
            else:
                us_second=_beats(opponent_last,last_ours,
                                opponent_last//13,trump)
            win_first+=mass*us_first
            win_second+=mass*us_second
            win_both+=mass*(us_first and us_second)
    return win_first,win_second,win_both


@lru_cache(maxsize=4096)
def _pair_probs_cached(x, y, pool, trump, actor_leads):
    return _two_player_pair_outcomes(x,y,pool,trump,actor_leads)


def _joint_remaining_trick_distribution(x, y, opponents, trump, leading):
    """(P0,P1,P2) with exact 2P pairing and 3P+ marginal-factor approximation.

    For multi-opponent rounds, first and second winning marginals are
    products of each independent two-player marginal. The joint two-win
    probability is the product of per-opponent two-win probabilities.
    This is a tractable *approximation*: with a different opponent winning
    trick one, the actual next leader differs from the duel's assumed leader.
    """
    first=second=both=1.0
    for pool in opponents:
        p1,p2,p12=_pair_probs_cached(x,y,pool,trump,leading)
        first*=p1
        second*=p2
        both*=p12
    p0=max(0.0, 1-first-second+both)
    p2=max(0.0,both)
    p1=max(0.0,1-p0-p2)
    total=p0+p1+p2
    return p0/total,p1/total,p2/total


def _score_distribution(p, needed):
    return p[1]+2*p[2]+(10*p[needed] if 0<=needed<=2 else 0)


def two_card_joint_value(a, b, needed, leading, opponents, trump):
    """Analytical expected score under a stochastic two-card order policy."""
    ab = _joint_remaining_trick_distribution(a,b,opponents,trump,leading)
    ba = _joint_remaining_trick_distribution(b,a,opponents,trump,leading)
    qa,qb = _score_distribution(ab,needed),_score_distribution(ba,needed)
    # Same softmax temperature as existing 2-card continuation.
    p=1/(1+exp((qb-qa)/_TEMPERATURE))
    return p*qa+(1-p)*qb


def analytical_three_card_values(snapshot, legal_cards):
    """Information-safe 3-card heuristic with joint last-2-trick evaluation.

    Accepts only an actor InformationState, not Game or true hands.
    Produces tuples of (legal card name, expected additive score).
    """
    if snapshot.phase!="playing" or snapshot.turn!=snapshot.observer_id:
        raise ValueError("Expected current playing-phase actor observation")
    own=snapshot.own_hand
    legal=tuple(legal_cards)
    if len(own)!=3 or len(legal)<2 or len(set(legal))!=len(legal):
        raise ValueError("Need three held cards and at least two legal actions")
    if not set(legal)<=set(own):
        raise ValueError("All options must be held")
    lead = snapshot.pile[0][1][-1] if snapshot.pile else None
    if lead is not None and any(c[-1]==lead for c in own):
        if any(c[-1]!=lead for c in legal):
            raise ValueError("Candidate violates follow-suit")
    # We approximate the unknown opponent 3->2 card depletion following
    # this current trick by a uniformly feasible two-card posterior.
    # Publicly known void suits are already removed from possible_cards.
    opponents=tuple(tuple(sorted(CARD_ID[c] for c in opp.possible_cards))
                    for opp in snapshot.opponents)
    needed=snapshot.bids[snapshot.observer_id]-snapshot.tricks_won[snapshot.observer_id]
    trump=SUITS.index(snapshot.trump)
    result=[]
    for card in legal:
        others=tuple(CARD_ID[c] for c in own if c!=card)
        a,b=others
        p=current_trick_win_probability(snapshot,CARD_ID[card])
        win_value=two_card_joint_value(a,b,needed-1,True,opponents,trump)
        lose_value=two_card_joint_value(a,b,needed,False,opponents,trump)
        result.append((card,p*(1+win_value)+(1-p)*lose_value))
    return tuple(result)


def select_analytical_three_card(snapshot, legal_cards, rng=random):
    values=analytical_three_card_values(snapshot,legal_cards)
    top=max(v for _,v in values)
    weights=[exp((v-top)/_TEMPERATURE) for _,v in values]
    stop=rng.random()*sum(weights)
    acc=0.0
    for (card,_),weight in zip(values,weights):
        acc+=weight
        if stop<acc:
            return card
    return values[-1][0]
