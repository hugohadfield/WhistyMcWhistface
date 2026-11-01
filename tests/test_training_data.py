"""Regression tests for information-safe trace generation and MC histogram labels."""
import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pytest

from alphawhist.core.FastRollout import evaluate_actions
from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistStructure import Game
from alphawhist.learning.encoding import FEATURE_DIM, encode_decision
from alphawhist.learning.recording import TrainingRecorder, iter_trace_decisions
from benchmarks.export_training_labels import (
    _label_one, _split_for_game, label_traces,
)
from benchmarks.generate_training_games import (
    generate, play_training_game,
)
from benchmarks.endgame_policy_tournament import play_match


def observation(cards=("14h", "2s", "8c"), nplayers=3):
    state = InformationState.new_round(
        cards, nplayers, round_index=0, dealer=nplayers - 1,
        leader=0, cards_per_player=len(cards),
    )
    state = state.apply_public_event({
        "type": "trump", "player": nplayers - 1, "suit": "h",
    })
    for p in range(nplayers):
        state = state.apply_public_event({
            "type": "bid", "player": p, "bid": 1,
        })
    return state


@pytest.mark.parametrize("nplayers", range(2, 8))
@pytest.mark.parametrize("workers,batch", [(1, 1), (1, 13), (2, 7)])
def test_histograms_match_unchanged_score_evaluator(nplayers, workers, batch):
    state = observation(nplayers=nplayers)
    standard = evaluate_actions(
        state, n_rollouts=37, seed=1955, workers=workers, batch_size=batch,
    )
    teacher = evaluate_actions(
        state, n_rollouts=37, seed=1955, workers=workers, batch_size=batch,
        collect_trick_histograms=True,
    )
    assert teacher.mean_round_scores == standard.mean_round_scores
    assert teacher.values == standard.values
    assert standard.additional_trick_counts is None
    assert len(teacher.additional_trick_counts) == len(state.own_hand)
    for hist, expected_score in zip(
            teacher.additional_trick_counts, teacher.values):
        assert len(hist) == 8 and sum(hist) == 37
        assert hist[4:] == (0, 0, 0, 0)
        recomputed = sum(
            count * (k + 10 * (k == state.bids[0]))
            for k, count in enumerate(hist)
        ) / 37
        assert recomputed == pytest.approx(expected_score)


def test_histogram_mid_trick_counts_include_current_partial_trick():
    state = observation()
    state = state.apply_public_event({
        "type": "card", "player": 0, "card": "14h",
    })
    state = state.apply_public_event({
        "type": "card", "player": 1, "card": "3s",
    })
    state = state.apply_public_event({
        "type": "card", "player": 2, "card": "4h",
    })
    state = state.apply_public_event({
        "type": "trick_end", "winner": 0,
    })
    teacher = evaluate_actions(
        state, n_rollouts=27, seed=510,
        collect_trick_histograms=True,
    )
    for hist, value in zip(teacher.additional_trick_counts, teacher.values):
        assert sum(hist) == 27
        assert hist[3:] == (0, 0, 0, 0, 0)
        manual = sum(
            n * (1 + k + 10 * (k == 0)) for k, n in enumerate(hist)
        ) / 27
        assert value == pytest.approx(manual)


def test_record_full_game_and_privileged_deals_are_separate(tmp_path):
    path = tmp_path / "public.jsonl"
    private = tmp_path / "privileged.jsonl"
    random.seed(4390)
    game = Game("testConfigFiles/ThreePlayerRandom.json")
    with TrainingRecorder(path, "test-4390", metadata={
        "deal_group": "test-match", "seed": 4390,
    }, privileged_path=private) as rec:
        game.training_recorder = rec
        with contextlib.redirect_stdout(io.StringIO()):
            game.playFullGame()
        assert rec.decisions_written == 165
    public = [json.loads(line) for line in path.read_text().splitlines()]
    secrets = [json.loads(line) for line in private.read_text().splitlines()]
    decisions = [r for r in public if r["type"] == "decision"]
    assert len(decisions) == 165
    assert sum(r["type"] == "round_start" for r in public) == 13
    assert sum(r["type"] == "round_end" for r in public) == 13
    assert public[-1]["type"] == "game_end"
    assert len(secrets) == 13
    assert all("hands_by_seat" in item for item in secrets)
    assert all("hands_by_seat" not in item for item in public)
    for item in decisions:
        state = InformationState.from_dict(item["observation"])
        encoded = encode_decision(state)
        assert item["chosen_card"] in item["legal_cards"]
        assert item["deal_group"] == "test-match"
        assert item["actual_remaining_tricks_won"] + state.tricks_won[item["player"]] == item["final_tricks_won"]
        assert item["actual_round_score"] == (
            item["final_tricks_won"]
            + 10 * (item["final_tricks_won"] == state.bids[item["player"]])
        )
        assert "policy_state" in item["observation"]
        assert item["observation"]["policy_state"] is None
        assert encoded.legal_mask.sum() == len(item["legal_cards"])
        assert "real_hand" not in json.dumps(item["observation"])
    assert all(len(r["observation"]["public_history"]) <= 40 for r in decisions)


def test_neural_export_shards_and_group_split_are_reproducible(tmp_path):
    traces = tmp_path / "traces"
    generate(games=2, seed=20135, players=3,
             agents=("random",), workers=2, output_dir=traces)
    paths = sorted(traces.glob("*.jsonl"))
    assert len(paths) == 2
    all_items = list(iter_trace_decisions(paths, include_forced=True))
    assert len(all_items) == 330
    nonforced = list(iter_trace_decisions(paths))
    assert 0 < len(nonforced) < 330

    item = _label_one(nonforced[0], 13, 1, 77, "random", 4)
    assert item["features"].shape == (FEATURE_DIM,)
    assert item["trick_counts"].shape == (52, 8)
    assert int(item["trick_counts"].sum()) == (
        13 * len(nonforced[0]["legal_cards"])
    )
    first = _label_one(nonforced[0], 13, 2, 77, "random", 3)
    np.testing.assert_array_equal(item["trick_counts"], first["trick_counts"])

    out = tmp_path / "labels"
    manifest = label_traces(paths, out, rollouts=13, workers=2,
                            seed=77, policy="random", batch_size=5,
                            shard_size=4, max_states=10)
    assert manifest["total_states"] == 10
    assert sum(manifest["counts_by_split"].values()) == 10
    assert (out / "manifest.json").exists()
    files = sorted(out.glob("*.npz"))
    assert files
    for path in files:
        with np.load(path) as data:
            assert data["features"].shape[1] == FEATURE_DIM
            assert data["legal_mask"].shape[1] == 52
            assert data["trick_counts"].shape[1:] == (52, 8)
            for i in range(len(data["features"])):
                valid = data["legal_mask"][i]
                assert all(int(row.sum()) == 13
                           for row in data["trick_counts"][i][valid])
                assert not data["trick_counts"][i][~valid].any()
    with pytest.raises(FileExistsError):
        label_traces(paths, out, rollouts=8, max_states=1)
    assert _split_for_game(77, {
        **nonforced[0], "game_id": "other",
        "deal_group": nonforced[0]["deal_group"],
    }) == _split_for_game(77, nonforced[0])


def test_tournament_can_export_learning_traces(tmp_path):
    match = play_match(0, seed=20262399, rollouts=8,
                       training_dir=tmp_path)
    path = tmp_path / "match-0000000.jsonl"
    assert path.exists()
    decisions = list(iter_trace_decisions([path], include_forced=True))
    assert len(decisions) == 165
    assert all(d["game_id"].startswith("tournament-") for d in decisions)
    assert match.game == 1


def test_parallel_training_games_reproduce_all_data(tmp_path):
    a, b = tmp_path / "one", tmp_path / "two"
    generate(games=2, seed=20300, players=3,
             agents=("random",), workers=1, output_dir=a)
    generate(games=2, seed=20300, players=3,
             agents=("random",), workers=2, output_dir=b)
    assert all((a / name).read_bytes() == (b / name).read_bytes()
               for name in ("game-0000000.jsonl", "game-0000001.jsonl"))
    with pytest.raises(FileExistsError):
        play_training_game(0, seed=20300, output_dir=a)
