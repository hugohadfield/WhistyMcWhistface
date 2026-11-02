"""Bid-aware 3-card heuristic: legality, no leakage, recursion, tournament."""
import contextlib
import io
import math
import random
from types import SimpleNamespace

import pytest

from BidAware2Player import BidAware2Player
from BidAware3Player import BidAware3Player
from MonteCarloPlayer import MonteCarloPlayer
from RandomPlayer import RandomPlayer
from alphawhist.core.EndgamePolicy import (
    _two_card_continuation_value, pick_bid_aware_two_card,
    pick_bid_aware_three_card, three_card_values,
)
from alphawhist.core.FastRollout import (
    CARDS, CARD_ID, SUITS, hand_mask, SimState, finish_round,
)
from alphawhist.core.WhistStructure import Game
from benchmarks.bid_aware_3_tournament import play_match, summarize, tournament


def sample_state(players=3, cards=("14h", "8s", "4d"), me=0, bid=1,
                 won=0, pile=()):
    hands=[0]*players
    hands[me]=hand_mask(cards)
    bids=[0]*players
    bids[me]=bid
    tricks=[0]*players
    tricks[me]=won
    return SimpleNamespace(hands=hands, turn=me,
                           pile=[CARD_ID[c] for c in pile],
                           bids=tuple(bids), tricks_won=tricks, trump=0)


def game_with_hands(players=3, cards=3, aware_seat=1):
    game = Game.__new__(Game)
    game.cardnumbers=[cards]
    game.playerConfigFileName=""
    game.numberofplayers=players
    game.players=[BidAware3Player(i) if i==aware_seat else RandomPlayer(i)
                  for i in range(players)]
    game.resetGame()
    return game


def test_three_card_values_finite_and_sensitive_to_bid():
    state=sample_state()
    legal=state.hands[0]
    cards,values=three_card_values(state,legal)
    assert set(cards)==set(CARD_ID[c] for c in ("14h","8s","4d"))
    assert all(math.isfinite(v) and 0<=v<=13 for v in values)
    assert three_card_values(sample_state(bid=0), legal)[1] != values
    assert three_card_values(sample_state(bid=3), legal)[1] != values
    assert pick_bid_aware_three_card(state,legal,random.Random(55)) in cards


@pytest.mark.parametrize("players", range(2,8))
@pytest.mark.parametrize("bid", (0,1,2,3))
def test_three_card_choice_only_dependent_on_own_cards_public_state(players,bid):
    me=players-1
    state=sample_state(players=players,me=me,bid=bid)
    legal=state.hands[me]
    chosen=pick_bid_aware_three_card(state,legal,random.Random(271828))
    assert legal & (1<<chosen)
    # Other players' true cards cannot affect the heuristic. Intentionally
    # provide *inconsistent* hidden hands to catch accidental access.
    state.hands=[hand_mask(("2h","3h","4h")) if p!=me else legal
                 for p in range(players)]
    assert pick_bid_aware_three_card(state,legal,random.Random(271828))==chosen


def test_two_card_policy_unchanged_for_same_seed():
    state=sample_state(cards=("14h","8s"))
    legal=state.hands[0]
    for seed in range(30):
        assert pick_bid_aware_three_card(state,legal,random.Random(seed)) == (
            pick_bid_aware_two_card(state,legal,random.Random(seed)))
    assert three_card_values(state,legal) is None


def test_forced_follow_suit_and_random_fallback():
    state=sample_state(pile=("12d",))
    legal=hand_mask(("4d",))
    assert three_card_values(state,legal) is None
    assert pick_bid_aware_three_card(state,legal,random.Random(3))==CARD_ID["4d"]
    state=sample_state(cards=("14h","8s","4d","5c"))
    mask=state.hands[0]
    for seed in range(10):
        assert pick_bid_aware_three_card(state,mask,random.Random(seed)) == (
            pick_bid_aware_two_card(state,mask,random.Random(seed)))


def test_two_card_continuation_models_win_loss_lead_change():
    a,b=CARD_ID["14h"],CARD_ID["3c"]
    for leading in (True,False):
        for needed in (-1,0,1,2,3):
            value=_two_card_continuation_value(a,b,needed,leading,0,2)
            assert 0<=value<=12
    assert (_two_card_continuation_value(a,b,1,True,0,2)
            != _two_card_continuation_value(a,b,1,False,0,2))


def test_real_player_matches_exact_simulated_three_card_choice():
    game=game_with_hands()
    game.players[0].set_hands(["2c","3h","4c"])
    game.players[1].set_hands(["14h","8s","7d"])
    game.players[2].set_hands(["5d","6c","9s"])
    game.resetRound()
    game._begin_observer_round(3)
    game.trumpsuit="h"
    game._record_public_event({"type":"trump","player":game.dealer,"suit":"h"})
    with contextlib.redirect_stdout(io.StringIO()):
        game.getPlayerBids()
    game.turn=game.leader  # dealer=0 => player 1 starts
    observer=game.observe(1)
    assert observer.turn==1
    mask=hand_mask(observer.own_hand)
    state=SimpleNamespace(
        hands=(0,mask,0),turn=1,pile=[],bids=observer.bids,
        tricks_won=observer.tricks_won,trump=SUITS.index("h"))
    random.seed(1923)
    expected=CARDS[pick_bid_aware_three_card(state,mask,random)]
    random.seed(1923)
    with contextlib.redirect_stdout(io.StringIO()):
        actual=game.players[1].makeMove([],"h",game)
    assert actual==expected
    assert actual not in game.players[1].real_hand
    assert game.players[1].three_card_choices==1


@pytest.mark.parametrize("players", range(2,8))
def test_complete_real_game_with_3_and_2_card_play(players):
    game=game_with_hands(players=players,cards=4,aware_seat=players-1)
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber==1
    assert sum(game.tricksWon)==4
    assert BidAware3Player.makeBid is RandomPlayer.makeBid
    assert BidAware3Player.pick_trumps is RandomPlayer.pick_trumps
    assert issubclass(BidAware3Player, BidAware2Player)


def test_fast_rollout_policy_supports_three_cards_and_terminates():
    hands=[hand_mask(("14h","8c","7d")),
           hand_mask(("2h","4s","6c")),
           hand_mask(("9d","10s","3c"))]
    state=SimState(hands,[],(1,1,1),[0,0,0],leader=0,turn=0,trump=0)
    scores=finish_round(state,random.Random(331),policy="bid_aware_3")
    assert sum(state.tricks_won)==3
    assert len(scores)==3
    assert MonteCarloPlayer(
        0,additionalParameters={"monteCarloPolicy":"bid_aware_3"}
    ).monteCarloPolicy=="bid_aware_3"


def test_complete_matched_seat_tournament_and_files(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        games,report=tournament(
            baseline="aware2",players=3,games=6,workers=1,seed=8299,
            bootstrap=100,progress_every=0,json_path=tmp_path/"a.json",
            csv_path=tmp_path/"a.csv")
    assert len(games)==6
    assert [r.challenger_seat for r in games]==[0,1,2,0,1,2]
    assert len({r.deal_seed for r in games})==2
    assert report["independent_deals"]==2
    assert report["three_card_choices_per_game"]>=0
    assert report["challenger_first_place_share"] + 2*report["baseline_first_place_share_per_player"]==pytest.approx(1)
    assert report["challenger_exact_bid_rate"]+report["challenger_undertrick_rate"]+report["challenger_overtrick_rate"]==pytest.approx(1)
    assert (tmp_path/"a.json").is_file()
    assert (tmp_path/"a.csv").is_file()


def test_two_player_and_random_baselines_with_complete_rotations():
    matches=[play_match(i,players=2,baseline="random",seed=9163)
             for i in range(2)]
    assert [m.challenger_seat for m in matches]==[0,1]
    r=summarize(matches,seed=12,bootstrap=100)
    assert r["independent_deals"]==1
    assert r["neutral_first_place_share"]==.5


def test_reject_incomplete_rotations():
    with pytest.raises(ValueError,match="complete"):
        summarize([play_match(0,players=3,baseline="aware2")],bootstrap=100)
