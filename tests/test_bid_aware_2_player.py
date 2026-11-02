"""The real two-card player must match the cheap simulated policy exactly."""
import contextlib
import io
import random
from types import SimpleNamespace

import pytest

from alphawhist.core.FastRollout import CARD_ID, CARDS, SUITS, hand_mask
from alphawhist.core.EndgamePolicy import pick_bid_aware_two_card
from alphawhist.core.WhistStructure import Game
from BidAware2Player import BidAware2Player
from RandomPlayer import RandomPlayer
from benchmarks.bid_aware_2_vs_random import play_match, summarize, tournament


def make_playable(n=3, *, seat=0, cards=2):
    game=Game.__new__(Game)
    game.cardnumbers=[cards]
    game.playerConfigFileName=""
    game.numberofplayers=n
    game.players=[BidAware2Player(i) if i==seat else RandomPlayer(i)
                  for i in range(n)]
    game.resetGame()
    return game


def test_bid_aware_inherits_same_trump_and_bidding_as_random():
    assert BidAware2Player.pick_trumps is RandomPlayer.pick_trumps
    assert BidAware2Player.makeBid is RandomPlayer.makeBid


@pytest.mark.parametrize("n,seat", [(2,0),(2,1),(3,0),(3,1),(3,2),
                                     (4,3),(5,2),(6,5),(7,4)])
def test_real_game_two_card_player_only_reads_own_hand(n,seat,monkeypatch):
    # Each player receives a legal two-card hand from an independent deal.
    game=make_playable(n, seat=seat)
    observed=[]
    import BidAware2Player as module
    original=module.pick_bid_aware_two_card

    def spy(state, legal, rng):
        assert state.turn==seat
        assert len(state.hands)==n
        assert all(state.hands[p]==0 for p in range(n) if p!=seat)
        assert state.hands[seat].bit_count()==2
        assert legal.bit_count()==2
        observed.append((state, legal))
        return original(state,legal,rng)

    monkeypatch.setattr(module, "pick_bid_aware_two_card", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        game.playFullGame()
    assert game.roundnumber==1
    assert sum(game.tricksWon)==2
    assert observed  # the chosen seat leads with two legal cards, or follows void


def test_exact_heursitic_choice_from_same_public_state_and_rng(monkeypatch):
    game=make_playable(3,seat=1)
    game.players[0].set_hands(["2c","4d"])
    game.players[1].set_hands(["14h","8s"])
    game.players[2].set_hands(["5c","6d"])
    game._begin_observer_round(2)
    game.trumpsuit="h"
    game._record_public_event({"type":"trump","player":game.dealer,"suit":"h"})
    game.getPlayerBids()
    game.turn=game.leader
    # Actor P1 leads the first trick with two legal alternatives.
    player=game.players[1]
    snapshot=game.observe(1)
    assert snapshot.turn==1
    ownmask=hand_mask(snapshot.own_hand)
    expected=SimpleNamespace(
        hands=(0,ownmask,0),pile=[],bids=snapshot.bids,
        tricks_won=snapshot.tricks_won,trump=SUITS.index("h"),turn=1)
    random.seed(121)
    wanted=CARDS[pick_bid_aware_two_card(expected,ownmask,random)]
    random.seed(121)
    with contextlib.redirect_stdout(io.StringIO()):
        actual=player.makeMove([],"h",game)
    assert actual==wanted
    assert actual not in player.real_hand


def test_one_player_vs_one_random_full_matched_deal_and_summary():
    matches=[play_match(i,players=2,seed=1234) for i in range(2)]
    assert [m.aware_seat for m in matches]==[0,1]
    assert matches[0].deal_seed==matches[1].deal_seed
    summary=summarize(matches,seed=1234,bootstrap=100)
    assert summary["games"]==2
    assert summary["independent_deals"]==1
    assert 0<=summary["aware_first_place_share"]<=1
    assert summary["aware_bid_hit_rate"]<=1
    assert summary["aware_overtrick_rate"]+summary["aware_undertrick_rate"]+summary["aware_bid_hit_rate"]==pytest.approx(1)


def test_three_player_rotation_and_report_files(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        matches,summary=tournament(
            players=3,games=3,workers=1,bootstrap=100,
            seed=3213,progress_every=0,
            csv_path=tmp_path/"results.csv",
            json_path=tmp_path/"results.json")
    assert len(matches)==3
    assert [x.aware_seat for x in matches]==[0,1,2]
    assert len({x.deal_seed for x in matches})==1
    assert summary["independent_deals"]==1
    assert summary["aware_first_place_share"]+2*summary["random_first_place_share_per_player"]==pytest.approx(1)
    assert summary["random_only_expected_first_place_share"]==pytest.approx(1/3)
    assert (tmp_path/"results.csv").exists()
    assert (tmp_path/"results.json").exists()


def test_invalid_tournaments_and_incomplete_seat_groups():
    with pytest.raises(ValueError,match="multiple"):
        tournament(players=3,games=4,workers=1)
    with pytest.raises(ValueError,match="complete"):
        summarize([play_match(0,players=3)],bootstrap=100)
