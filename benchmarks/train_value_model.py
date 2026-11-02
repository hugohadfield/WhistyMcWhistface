"""Train the 52-card, eight-bin distributional Whist value MLP.

Uses the exporter-assigned whole-deal train/val/test splits. Reads one .npz
shard at a time, so a multi-million-row dataset need not fit in RAM. Selects
the best checkpoint using validation action cross-entropy only. Test data is
never used for model selection.

Example:
 python -m benchmarks.train_value_model --dataset data/labeled \
   --output-dir runs/whist-v1 --device auto --epochs 30 --batch-size 1024
"""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from alphawhist.learning.checkpoint import choose_device, save_checkpoint
from alphawhist.learning.dataset import (
    read_manifest, shard_batches, split_files, to_tensors,
)
from alphawhist.learning.metrics import evaluate_model
from alphawhist.learning.model import DistributionalWhistMLP, distributional_loss


def train(dataset, output_dir, *, epochs=30, batch_size=512,
          hidden_size=128, residual_blocks=2, lr=3e-4, weight_decay=1e-4,
          patience=8, seed=20261101, device="auto", cpu_threads=2):
    if (epochs < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0
            or patience < 1 or cpu_threads < 1):
        raise ValueError("Invalid training hyperparameters")
    manifest = read_manifest(dataset)
    train_files = split_files(dataset, manifest, "train")
    val_files = split_files(dataset, manifest, "val")
    # Require a genuine held-out test split, but don't look at its values.
    split_files(dataset, manifest, "test")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Training output directory must be empty")
    dev = choose_device(device)
    if dev.type == "cpu":
        torch.set_num_threads(cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DistributionalWhistMLP(hidden_size, residual_blocks).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    config = {
        "dataset": str(dataset),
        "teacher_policy": manifest["teacher_policy"],
        "teacher_rollouts": manifest.get("rollouts_per_action"),
        "device": str(dev), "epochs": epochs, "batch_size": batch_size,
        "hidden_size": hidden_size, "residual_blocks": residual_blocks,
        "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "seed": seed, "cpu_threads": cpu_threads,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (out / "training-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best_nll = float("inf")
    best_epoch = -1
    rows = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_ce = 0.
        legal_actions = 0
        seen_states = 0
        for batch in shard_batches(train_files, batch_size=batch_size,
                                   seed=seed + epoch):
            t = to_tensors(batch, dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(t["features"])
            loss = distributional_loss(
                logits, t["targets"], t["legal_mask"], t["remaining_tricks"]
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(t["legal_mask"].sum().item())
            weighted_ce += float(loss.item()) * count
            legal_actions += count
            seen_states += len(batch["features"])
        if not legal_actions:
            raise ValueError("Training split contains no legal actions")
        validation = evaluate_model(model, val_files, batch_size=batch_size,
                                    device=dev)
        val_ce = validation["action_cross_entropy"]
        row = {
            "epoch": epoch,
            "train_action_cross_entropy": weighted_ce / legal_actions,
            "train_states": seen_states,
            **{"val_" + k: v for k, v in validation.items()},
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(f"EPOCH {epoch} train_ce={row['train_action_cross_entropy']:.6f} "
              f"val_ce={val_ce:.6f} "
              f"val_score_mae={validation['expected_score_mae']:.4f} "
              f"val_regret={validation['teacher_expected_regret_per_decision']:.4f}",
              flush=True)
        if val_ce < best_nll - 1e-8:
            best_nll = val_ce
            best_epoch = epoch
            save_checkpoint(out / "best.pt", model, metadata={
                "epoch": epoch, "val_action_cross_entropy": val_ce,
                "teacher_policy": manifest["teacher_policy"],
                "dataset_schema": manifest["dataset_schema"],
                "training_seed": seed,
            })
        if epoch - best_epoch >= patience:
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    save_checkpoint(out / "last.pt", model, metadata={
        "epoch": rows[-1]["epoch"],
        "val_action_cross_entropy": rows[-1]["val_action_cross_entropy"],
        "teacher_policy": manifest["teacher_policy"],
        "training_seed": seed,
    })
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = {
        "best_epoch": best_epoch, "best_val_action_cross_entropy": best_nll,
        "epochs_completed": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": str(out / "best.pt"),
        "last_checkpoint": str(out / "last.pt"),
        "parameter_count": config["parameter_count"],
    }
    (out / "result.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("TRAIN_DONE " + json.dumps(results, sort_keys=True), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, epochs=args.epochs,
          batch_size=args.batch_size, hidden_size=args.hidden_size,
          residual_blocks=args.residual_blocks, lr=args.lr,
          weight_decay=args.weight_decay, patience=args.patience,
          seed=args.seed, device=args.device, cpu_threads=args.cpu_threads)


if __name__ == "__main__":
    main()
