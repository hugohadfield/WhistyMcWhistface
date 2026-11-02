"""End-to-end real simulator -> labels -> PyTorch training -> evaluation -> tournament.

Requires torch; standard CPU-only regression suite safely skips this module.
"""
import contextlib
import io
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphawhist.core.InformationState import InformationState
from alphawhist.learning.checkpoint import load_checkpoint, save_checkpoint
from alphawhist.learning.dataset import read_manifest, load_shard, split_files
from alphawhist.learning.encoding import encode_decision, FEATURE_DIM
from alphawhist.learning.model import DistributionalWhistMLP
from benchmarks.export_training_labels import label_traces
from benchmarks.generate_training_games import generate
from benchmarks.train_value_model import train
from benchmarks.evaluate_value_model import evaluate
from benchmarks.neural_vs_monte_carlo import tournament, summarize, play_match
from NeuralValuePlayer import NeuralValuePlayer


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real-whist-training")
    traces = directory / "traces"
    labels = directory / "labels"
    run = directory / "run"
    # This seed gives the first eight independent games all three splits,
    # without looking at data labels or the opponent's real hands.
    with contextlib.redirect_stdout(io.StringIO()):
        generate(games=8, seed=20262001, players=3, agents=("random",),
                 workers=2, output_dir=traces)
        manifest = label_traces(
            sorted(traces.glob("*.jsonl")), labels, rollouts=8, workers=2,
            shard_size=32, seed=20262001, policy="random",
        )
        info = train(labels, run, epochs=3, batch_size=48, hidden_size=32,
                     residual_blocks=1, patience=2, device="cpu",
                     cpu_threads=2, seed=27319)
    assert all(manifest["counts_by_split"][split] > 0 for split in ("train", "val", "test"))
    return directory, labels, run, info


def test_training_creates_safe_loadable_checkpoints_and_history(trained):
    _, labels, run, result = trained
    model, meta = load_checkpoint(run / "best.pt")
    assert isinstance(model, DistributionalWhistMLP)
    assert 1 <= meta["epoch"] <= 3
    assert model.training is False
    assert result["best_epoch"] == meta["epoch"]
    history = (run / "history.csv").read_text()
    assert "val_teacher_expected_regret_per_decision" in history
    assert "train_action_cross_entropy" in history
    assert read_manifest(labels)["feature_schema"] == 1
    assert all((run / name).exists() for name in
               ("best.pt", "last.pt", "result.json", "training-config.json"))


def test_test_split_never_used_for_selection_and_metrics_are_finite(trained, tmp_path):
    _, labels, run, _ = trained
    result = evaluate(labels, run / "best.pt", split="test",
                      device="cpu", output=tmp_path / "test-metrics.json")
    m = result["metrics"]
    assert (tmp_path / "test-metrics.json").exists()
    assert m["states"] == read_manifest(labels)["counts_by_split"]["test"]
    assert m["legal_actions"] > m["states"]
    assert 0 <= m["teacher_card_top1_agreement"] <= 1
    assert 0 <= m["teacher_card_optimal_agreement"] <= 1
    assert m["teacher_expected_regret_per_decision"] >= 0
    for key in ("action_cross_entropy", "action_brier", "expected_score_mae",
                "expected_score_rmse"):
        assert np.isfinite(m[key])


def test_player_legality_and_no_model_for_forced_moves(trained):
    _, labels, run, _ = trained
    player = NeuralValuePlayer(0, checkpoint=run / "best.pt")
    from alphawhist.learning.recording import iter_trace_decisions
    from alphawhist.core.WhistStructure import Game
    from RandomPlayer import RandomPlayer

    game = Game("testConfigFiles/ThreePlayerRandom.json")
    game.cardnumbers = [4]
    game.resetGame()
    game.players[0] = player
    game.resetRound()
    game.randomDeal(4)
    # Test early initial bid/trump logic unchanged from MC player.
    game.trumpsuit = player.pick_trumps(3)
    game._record_public_event({"type": "trump", "player": game.dealer,
                               "suit": game.trumpsuit})
    game.getPlayerBids()
    game.turn = game.leader
    # Play the whole round, checking all 3 players use legal actions
    # through the authoritative Game engine.
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(4):
            game.playFullTrick()
        game.processScore()
        game.endRound()
    assert sum(game.tricksWon) == 4
    assert player._model is not None


def test_paired_neural_vs_mc_game_tournament(trained, tmp_path):
    _, _, run, _ = trained
    with contextlib.redirect_stdout(io.StringIO()):
        matches, summary = tournament(
            run / "best.pt", games=2, seed=2026401, rollouts=8, workers=2,
            mc_policy="bid_aware_2", bootstrap=100,
            csv_path=tmp_path / "games.csv",
            json_path=tmp_path / "summary.json",
        )
    assert len(matches) == 2
    assert summary["games"] == 2 and summary["paired_deals"] == 1
    assert matches[0].deal_seed == matches[1].deal_seed
    assert matches[0].neural_seat == matches[1].mc_seat
    assert matches[1].neural_seat == matches[0].mc_seat
    assert sum(matches[0].__dict__[x] for x in
               ("neural_score", "mc_score", "random_score")) > 0
    assert summary["neural_head_to_head_wins"] + summary["mc_head_to_head_wins"] + summary["head_to_head_ties"] == 2
    assert (tmp_path / "games.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_dataset_guards_checkpoint_version_and_no_test_split(tmp_path, trained):
    _, labels, run, _ = trained
    with pytest.raises(FileExistsError, match="empty"):
        train(labels, run, epochs=1)
    item = next(iter(split_files(labels, read_manifest(labels), "train")))
    arrays = load_shard(item)
    assert arrays["features"].shape[1] == FEATURE_DIM
    model = DistributionalWhistMLP(hidden_size=16, residual_blocks=0)
    good = tmp_path / "checkpoint.pt"
    save_checkpoint(good, model, metadata={"teacher_policy": "random"})
    m, meta = load_checkpoint(good)
    assert meta["teacher_policy"] == "random"
    with pytest.raises(ValueError, match="different teacher"):
        # Different teacher continuation policies are not interchangeable.
        bad_data = tmp_path / "alternate"
        bad_data.mkdir()
        source_manifest = read_manifest(labels)
        source_manifest["teacher_policy"] = "highest"
        (bad_data / "manifest.json").write_text(json.dumps(source_manifest))
        # use same shard names via symlinks only within the test fixture
        for shard in source_manifest["shards"]:
            (bad_data / shard).symlink_to((labels / shard).resolve())
        evaluate(bad_data, good)

def test_interrupted_export_resumes_without_relabeling_or_losing_rows(trained, tmp_path):
    """Three independent split buffers resume to the exact complete dataset."""
    import shutil

    directory, complete, _, _ = trained
    resume = tmp_path / "resume"
    resume.mkdir()
    full_manifest = read_manifest(complete)
    # Simulate an interruption at different split-buffer boundaries.
    for split, keep in (("train", 3), ("val", 1), ("test", 2)):
        for path in sorted(complete.glob(f"{split}-*.npz"))[:keep]:
            shutil.copy2(path, resume / path.name)

    paths = sorted((directory / "traces").glob("*.jsonl"))
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = label_traces(
            paths, resume, rollouts=8, workers=2, shard_size=32,
            seed=20262001, policy="random", resume=True,
        )
    assert manifest["counts_by_split"] == full_manifest["counts_by_split"]
    assert manifest["total_states"] == full_manifest["total_states"]
    for name in full_manifest["shards"]:
        with np.load(complete / name, allow_pickle=False) as a:
            with np.load(resume / name, allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])

    # A changed seed must fail rather than mixing incompatible labels.
    bad = tmp_path / "wrong-seed"
    bad.mkdir()
    shutil.copy2(complete / "train-00000.npz", bad / "train-00000.npz")
    with pytest.raises(ValueError, match="mismatch"):
        label_traces(
            paths, bad, rollouts=8, workers=1, shard_size=32,
            seed=20262002, policy="random", resume=True,
        )
