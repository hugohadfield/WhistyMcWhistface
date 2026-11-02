"""Evaluate a trained Whist value MLP against untouched MC teacher labels.

Use split=test only after model selection; train uses val for early stopping.
Metrics include distributional cross entropy/Brier score, score MAE/RMSE,
top-card agreement and teacher-estimated decision regret. These are *teacher*
metrics, not actual tournament win rates.

Example:
 python -m benchmarks.evaluate_value_model --dataset data/labeled \
     --checkpoint runs/whist-v1/best.pt --split test --output runs/test.json
"""
import argparse
import json
from pathlib import Path

from alphawhist.learning.checkpoint import choose_device, load_checkpoint
from alphawhist.learning.dataset import read_manifest, split_files
from alphawhist.learning.metrics import evaluate_model


def evaluate(dataset, checkpoint, *, split="test", device="auto",
             batch_size=1024, output=None):
    manifest = read_manifest(dataset)
    files = split_files(dataset, manifest, split)
    dev = choose_device(device)
    model, metadata = load_checkpoint(checkpoint, dev)
    if (metadata.get("teacher_policy") is not None
            and metadata["teacher_policy"] != manifest["teacher_policy"]):
        raise ValueError("Checkpoint was trained with a different teacher continuation policy")
    metrics = evaluate_model(model, files, batch_size=batch_size, device=dev)
    report = {
        "checkpoint": str(checkpoint), "dataset": str(dataset),
        "split": split, "device": str(dev),
        "teacher_policy": manifest["teacher_policy"],
        "rollouts_per_action": manifest["rollouts_per_action"],
        "checkpoint_metadata": metadata, "metrics": metrics,
        "interpretation": "Teacher-relative predictions, NOT actual gameplay strength",
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("EVALUATION " + json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.dataset, args.checkpoint, split=args.split,
             device=args.device, batch_size=args.batch_size,
             output=args.output)


if __name__ == "__main__":
    main()
