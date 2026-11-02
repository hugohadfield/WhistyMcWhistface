"""Integration and matched-deal validity tests for MC3-vs-MC2-vs-random."""
import contextlib
import io
import json

import pytest

from alphawhist.core.InformationState import InformationState
from alphawhist.core.FastRollout import evaluate_actions
from benchmarks.mc_aware3_aware2_random import (
    ROLE_TO_SEAT, ROLE_NAMES, play_match, summarize, tournament,
)


def test_all_six_permutations_unique_and_role_balanced():
    assert len(ROLE_TO_SEAT)==6
    assert set(ROLE_TO_SEAT)==set(__import__("itertools").permutations(range(3)))
    assert len(ROLE_NAMES)==3
    for role in range(3):
        assert sorted(mapping[role] for mapping in ROLE_TO_SEAT)==[0,0,1,1,2,2]


def test_public_mc_api_actually_supports_bid_aware_three():
    # The old API erroneously rejected the policy before running any rollouts.
    s=InformationState.new_round(
        own_hand=("14h","10s","4d"),num_players=3,
        round_index=0,dealer=2,leader=0,cards_per_player=3,
        observer_id=0)
    s=s.apply_public_event({"type":"trump","player":2,"suit":"h"})
    for player in range(3):
        s=s.apply_public_event({"type":"bid","player":player,"bid":1})
    evaluation=evaluate_actions(s,("14h","10s","4d"),n_rollouts=8,
                                policy="bid_aware_3",seed=1701)
    assert evaluation.rollouts==8
    assert len(evaluation.values)==3
    assert evaluation.best_action() in s.own_hand


def test_one_real_full_game_and_outcomes():
    with contextlib.redirect_stdout(io.StringIO()):
        match=play_match(0,seed=321,rollouts=4)
    assert match.seats==(0,1,2)
    assert all(0<=count<=13 for count in match.exact_bids)
    assert all(e+o+u==13 for e,o,u in zip(
        match.exact_bids,match.overtricks,match.undertricks))
    assert sum(match.first_place_shares)==pytest.approx(1)


def test_six_rotations_same_deal_and_95_percent_ci(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        matches,stats=tournament(
            games=6,rollouts=4,seed=7192,workers=1,
            bootstrap=100,progress_every=0,
            json_path=tmp_path/"stats.json",
            csv_path=tmp_path/"results.csv")
    assert len(matches)==6
    assert [r.seats for r in matches]==list(ROLE_TO_SEAT)
    assert len({r.deal_seed for r in matches})==1
    assert all(r.deal_group==0 for r in matches)
    assert stats["independent_deals"]==1
    assert stats["rollouts_per_mc_card_decision"]==4
    assert stats["mc_search_objective"]=="expected own round points"
    assert sum(stats["first_place_shares_by_role"].values())==pytest.approx(1)
    assert stats["mc3_minus_mc2_mean_score"]==pytest.approx(
        sum(r.scores[0]-r.scores[1] for r in matches)/6)
    for role in ROLE_NAMES:
        assert stats["exact_bid_rates_by_role"][role] + (
            stats["overtrick_rates_by_role"][role] +
            stats["undertrick_rates_by_role"][role])==pytest.approx(1)
    assert (tmp_path/"stats.json").is_file()
    assert len((tmp_path/"results.csv").read_text().splitlines())==7
    assert json.loads((tmp_path/"stats.json").read_text())["games"]==6


def test_summary_rejects_incomplete_groups():
    with contextlib.redirect_stdout(io.StringIO()):
        one=play_match(0,seed=99,rollouts=2)
    with pytest.raises(ValueError,match="complete"):
        summarize([one])
    with pytest.raises(ValueError,match="multiple of six"):
        tournament(games=5)
