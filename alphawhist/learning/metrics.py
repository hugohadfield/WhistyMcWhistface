"""Held-out metrics against per-action MC trick-count distributions.

All scores are computed using the actual 10-point exact-bid bonus. Metrics
are aggregated across batches (action-weighted CE/MAE, state-weighted regret)
and do not evaluate against the actual unseen opponents' hands.
"""
import torch

from alphawhist.learning.dataset import shard_batches, to_tensors


@torch.no_grad()
def evaluate_model(model, files, batch_size=512, device="cpu"):
    model.eval()
    totals = {
        "states": 0, "legal_actions": 0, "cross_entropy_sum": 0.0,
        "brier_sum": 0.0, "score_abs_sum": 0.0,
        "score_squared_sum": 0.0, "regret_sum": 0.0,
        "teacher_top1_agreements": 0, "teacher_optimal_agreements": 0,
    }
    for batch in shard_batches(files, batch_size=batch_size):
        t = to_tensors(batch, device)
        mask = t["legal_mask"]
        probs = model.predict(
            t["features"], mask, t["remaining_tricks"],
            t["tricks_won"], t["bids"],
        )
        target = t["targets"]
        action_count = int(mask.sum().item())
        states = len(mask)
        log_prob = probs.probabilities.clamp_min(1e-30).log()
        ce = -(target * log_prob).sum(-1)[mask].sum()
        squared = ((probs.probabilities - target) ** 2).sum(-1)[mask].sum()
        k = torch.arange(8, device=t["features"].device)[None, :]
        total_tricks = t["tricks_won"][:, None] + k
        scores = total_tricks + 10 * (total_tricks == t["bids"][:, None])
        q_teacher = (target * scores[:, None, :]).sum(-1)
        q_teacher = q_teacher.masked_fill(~mask, float("-inf"))
        q_pred = probs.expected_scores
        difference = q_pred[mask] - q_teacher[mask]
        teacher_best = q_teacher.argmax(-1)
        predicted_best = probs.best_card_ids()
        best_value = q_teacher.gather(1, teacher_best[:, None]).squeeze(1)
        picked_value = q_teacher.gather(1, predicted_best[:, None]).squeeze(1)
        totals["states"] += states
        totals["legal_actions"] += action_count
        totals["cross_entropy_sum"] += float(ce.item())
        totals["brier_sum"] += float(squared.item())
        totals["score_abs_sum"] += float(difference.abs().sum().item())
        totals["score_squared_sum"] += float(difference.square().sum().item())
        totals["regret_sum"] += float((best_value - picked_value).clamp_min(0).sum().item())
        totals["teacher_top1_agreements"] += int((teacher_best == predicted_best).sum().item())
        # Tie-aware optimality is more meaningful when MC estimates are tied.
        totals["teacher_optimal_agreements"] += int(
            ((best_value - picked_value).abs() <= 1e-6).sum().item()
        )
    n, a = totals["states"], totals["legal_actions"]
    if not n or not a:
        raise ValueError("Cannot evaluate empty held-out dataset")
    return {
        "states": n,
        "legal_actions": a,
        "mean_legal_actions": a / n,
        "action_cross_entropy": totals["cross_entropy_sum"] / a,
        "action_brier": totals["brier_sum"] / a,
        "expected_score_mae": totals["score_abs_sum"] / a,
        "expected_score_rmse": (totals["score_squared_sum"] / a) ** .5,
        "teacher_card_top1_agreement": totals["teacher_top1_agreements"] / n,
        "teacher_card_optimal_agreement": totals["teacher_optimal_agreements"] / n,
        "teacher_expected_regret_per_decision": totals["regret_sum"] / n,
    }
