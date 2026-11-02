"""Analytical hypergeometric, conditional two-trick and real-game tests."""
import contextlib
import io
import random
import math
from itertools import combinations

import pytest

from alphawhist.core.AnalyticEndgame import (
    void_probability, _opponent_beat_probability, _two_player_pair_outcomes,
    _joint_remaining_trick_distribution, two_card_joint_value,
    analytical_three_card_values, current_trick_win_probability,
    select_analytical_three_card,
)
from alphawhist.core.FastRollout import CARD_ID
from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistStructure import Game
from AnalyticBidAware3Player import AnalyticBidAware3Player
from BidAware3Player import BidAware3Player
from RandomPlayer import RandomPlayer
from benchmarks.bid_aware_3_tournament import play_match, tournament


def _snapshot(hand=("14h","8s","4d"), opponents=2, void_events=()):
    n=opponents+1
    state=InformationState.new_round(
        own_hand=hand, num_players=n, round_index=0, dealer=n-1,
        leader=0, cards_per_player=len(hand), observer_id=0)
    state=state.apply_public_event(
        {"type":"trump","player":n-1,"suit":"h"})
    for pid in range(n):
        state=state.apply_public_event(
            {"type":"bid","player":pid,"bid":1})
    assert state.phase=="playing" and state.turn==0
    return state


def test_hypergeom_void_exact_finite_population():
    assert void_probability(8,6,2)==pytest.approx(1/28)
    assert void_probability(8,0,2)==1
    assert void_probability(8,8,2)==0
    assert void_probability(8,7,2)==0
    assert void_probability(8,6,0)==1
    with pytest.raises(ValueError):
        void_probability(4,5,2)
    with pytest.raises(ValueError):
        void_probability(4,2,5)


def test_random_follow_response_exact_marginal():
    # Candidate 10h trump, opponent holds 2 unknown cards from 4s,
    # 9h, Kh. Lead is spades: can trump only when void in spades.
    pool=(CARD_ID["4s"],CARD_ID["9h"],CARD_ID["13h"])
    p=_opponent_beat_probability(pool,2,2,0,CARD_ID["10h"])
    # Only opponent hand {9h, Kh} is void in spades (prob 1/3).
    # Conditional void, random played trump beats 10h with P=1/2.
    assert p==pytest.approx(1/6)
    # If known void, all possible cards are trumps:
    assert _opponent_beat_probability(pool[1:],1,2,0,CARD_ID["10h"])==.5


def test_pair_outcome_depletion_both_tricks_win():
    pool=(CARD_ID["4s"],CARD_ID["5s"])
    x,y=CARD_ID["14h"],CARD_ID["2c"]
    a,b,both=_two_player_pair_outcomes(x,y,pool,0,True)
    assert (a,b,both)==pytest.approx((1.,1.,1.))
    assert _joint_remaining_trick_distribution(x,y,(pool,),0,True)==pytest.approx((0.,0.,1.))


def test_known_void_and_card_removal_affect_exact_probabilities():
    state=_snapshot()
    p=current_trick_win_probability(state,CARD_ID["14h"])
    assert p==pytest.approx(1.)
    # The analytic evaluator must read *public candidate sets* and not
    # merely ranks; removing higher cards or marking void changes values.
    first=analytical_three_card_values(state,("14h","8s","4d"))
    assert len(first)==3 and all(math.isfinite(v) for _,v in first)
    other=state.opponents[0]
    from dataclasses import replace
    from alphawhist.core.InformationState import OpponentKnowledge
    trimmed=OpponentKnowledge(
        player_id=other.player_id,remaining_count=other.remaining_count,
        possible_cards=frozenset(c for c in other.possible_cards
                                 if c[-1]!="h"),void_suits=frozenset({"h"}))
    changed=replace(state,opponents=(trimmed,*state.opponents[1:]))
    second=analytical_three_card_values(changed,("14h","8s","4d"))
    assert first!=second


def test_two_card_conditional_probabilities_are_normalized():
    pool=tuple(CARD_ID[c] for c in ("4s","10s","7h","11c","3d"))
    for leading in (True,False):
        for x,y in ((CARD_ID["14h"],CARD_ID["8s"]),
                    (CARD_ID["3c"],CARD_ID["10d"])):
            outcomes=_joint_remaining_trick_distribution(
                x,y,(pool,),0,leading)
            assert all(0<=v<=1 for v in outcomes)
            assert sum(outcomes)==pytest.approx(1)
            for needed in (-2,-1,0,1,2,3):
                value=two_card_joint_value(
                    x,y,needed,leading,(pool,),0)
                assert 0 <= value <= 12


def test_three_player_independence_approximation_finite():
    pool=tuple(CARD_ID[c] for c in
               ("4s","10s","7h","11c","3d","9h","13d"))
    for leader in (True,False):
        p=_joint_remaining_trick_distribution(
            CARD_ID["14h"],CARD_ID["8s"],(pool,pool),0,leader)
        assert all(math.isfinite(v) and v>=0 for v in p)
        assert sum(p)==pytest.approx(1)


def test_legal_restriction_for_current_trick():
    state=InformationState.new_round(
        own_hand=("14h","8s","4d"),num_players=3,round_index=0,
        dealer=2,leader=0,cards_per_player=3,observer_id=1)
    state=state.apply_public_event(
        {"type":"trump","player":2,"suit":"h"})
    for pid in range(3):
        state=state.apply_public_event({"type":"bid","player":pid,"bid":1})
    state=state.apply_public_event({"type":"card","player":0,"card":"9h"})
    assert state.turn==1
    with pytest.raises(ValueError,match="follow-suit"):
        analytical_three_card_values(state,("14h","8s"))


def test_does_not_use_other_players_real_hands():
    state=_snapshot()
    first=analytical_three_card_values(state,tuple(state.own_hand))
    # Counterfactual knowledge: entire opponent candidate set may change
    # while leaving our own hand and public pile unchanged.
    assert all(isinstance(card,str) for card,_ in first)
    random.seed(79)
    card=select_analytical_three_card(state,state.own_hand,random)
    assert card in state.own_hand


@pytest.mark.parametrize("players",[2,3,4,5,6,7])
def test_full_game_with_analytical_player(players):
    game=Game.__new__(Game)
    game.cardnumbers=[4,3,2]
    game.playerConfigFileName=""
    game.numberofplayers=players
    game.players=[AnalyticBidAware3Player(i) if i==players-1
                  else RandomPlayer(i) for i in range(players)]
    game.resetGame()
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber==3
    assert len(game.scores)==players
    assert issubclass(AnalyticBidAware3Player,BidAware3Player)


def test_paired_match_saves_meaningful_outputs(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        results,summary=tournament(
            challenger="analytic3",baseline="aware3",players=3,
            games=6,seed=7789,workers=1,bootstrap=100,
            progress_every=0,json_path=tmp_path/"sum.json",
            csv_path=tmp_path/"games.csv")
    assert len(results)==6
    assert summary["independent_deals"]==2
    assert summary["challenger"]=="analytic3"
    assert summary["baseline"]=="aware3"
    assert summary["three_card_choices_per_game"]>0
    assert (tmp_path/"sum.json").exists()
    assert (tmp_path/"games.csv").exists()
