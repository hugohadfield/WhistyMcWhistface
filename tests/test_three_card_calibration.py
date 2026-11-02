"""Regression and smoke tests for actual-trick and MC-reference diagnostics."""
import contextlib
import io
import json

import pytest

from alphawhist.core.EndgamePolicy import _THREE_CARD_WIN, _current_win_probability
from benchmarks.three_card_calibration import (
    all_probability_reports, all_ranking_reports, bucket_probability,
    empirical_probability_rows, probability_summary, rank_and_regret,
    read_decisions, reconstructed_state, run,
)
from benchmarks.generate_training_games import generate


def test_three_card_trace_records_actual_trick_outcomes_without_privileged_hands(tmp_path):
    directory=tmp_path/"traces"
    with contextlib.redirect_stdout(io.StringIO()):
        generate(games=2,players=3,agents=("aware3","random"),workers=1,
                 seed=220919,output_dir=directory)
    rows,count=read_decisions(directory)
    assert count==2 and rows
    assert all(len(x["record"]["observation"]["own_hand"])==3 for x in rows)
    assert all(len(x["record"]["legal_cards"])>=2 for x in rows)
    assert all(x["won_trick"] in (0,1) for x in rows)
    assert all(0<=x["position"]<3 for x in rows)
    assert all(x["record"]["behavior_policy"]=="BidAware3Player" for x in rows)
    assert all("opponent_real_hand" not in json.dumps(x) for x in rows)
    for item in rows:
        obs,state = reconstructed_state(item)
        assert obs.observer_id==item["player"]
        assert state.hands[item["player"]].bit_count()==3
        assert all(state.hands[p]==0 for p in range(3) if p!=item["player"])
        assert 0<=_current_win_probability(
            state,
            __import__("alphawhist.core.FastRollout",fromlist=["CARD_ID"]).CARD_ID[
                item["record"]["chosen_card"]],
            _THREE_CARD_WIN) <= 1
    probabilities=all_probability_reports(empirical_probability_rows(rows))
    assert probabilities["overall"]["n"]==len(rows)
    assert probabilities["by_players"]["3"]["n"]==len(rows)
    assert sum(x["n"] for x in probabilities["by_position"]["3"].values())==len(rows)


def test_probability_bins_include_p_equal_one_and_nondegenerate_brier():
    assert bucket_probability(0)==0
    assert bucket_probability(.999)==4
    assert bucket_probability(1)==4
    rows=[
        {"p_predicted":1.0,"won":1,"brier":0.0},
        {"p_predicted":.0,"won":0,"brier":0.0},
        {"p_predicted":.8,"won":0,"brier":.64},
    ]
    result=probability_summary(rows)
    assert result["n"]==3
    assert result["mean_brier"]==pytest.approx(.64/3)


def test_mc_ranking_all_legal_cards_on_same_world_and_smoke_run(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        result=run(
            output_dir=tmp_path/"audit",games_per_players=1,workers=1,
            max_states_per_player=3,rollouts=16,confirm_rollouts=32,
            confirm_top=1,seed=38292)
    meta=result["metadata"]
    assert meta["games"]["2"]["games"]==1
    assert meta["games"]["3"]["games"]==1
    assert meta["mc_states"]>=2
    assert len(result["confirmed"])==2
    assert all("fresh_regret" in r for r in result["confirmed"])
    assert (tmp_path/"audit"/"summary.json").exists()
    assert (tmp_path/"audit"/"analysis.md").exists()
    assert (tmp_path/"audit"/"decision-comparisons.jsonl").exists()
    assert "2" in result["mc_ranking"] and "3" in result["mc_ranking"]
