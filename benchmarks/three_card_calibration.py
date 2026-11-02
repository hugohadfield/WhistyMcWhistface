"""Investigate why three-card bid-aware play helps less with 3 players.

Collect only actor-visible information state snapshots from complete real
matches (BidAware3Player vs RandomPlayer). Measure calibration of the
heuristic's prediction that the ACTUALLY CHOSEN card takes the current trick,
against observed outcomes; then compare ALL legal three-card action rankings
to fresh, shared-hidden-deal high-budget MC counterfactual evaluations.

This diagnostic distinguishes calibration from action ranking; it does NOT
treat the MC continuation model as ground truth. No privileged replay data.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics

from alphawhist.core.EndgamePolicy import (
    _THREE_CARD_WIN, _current_win_probability, three_card_values,
)
from alphawhist.core.FastRollout import (
    CARD_ID, CARDS, SUITS, evaluate_actions, hand_mask, trick_winner,
)
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.recording import iter_trace_decisions
from benchmarks.generate_training_games import generate


def read_decisions(directory):
    """Read complete public decision traces, including forced plays for outcomes."""
    paths = sorted(directory.glob("game-*.jsonl"))
    if not paths:
        raise ValueError(f"No complete game traces in {directory}")
    selected, game_count = [], 0
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.strip()]
        if records[0]["type"] != "game_start" or records[-1]["type"] != "game_end":
            raise ValueError(f"Incomplete trace: {path}")
        game_count += 1
        decisions = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record["type"] == "decision":
                snapshot = record["observation"]
                decisions[record["round"]][snapshot["trick_number"]].append(record)
        for r, tricks in decisions.items():
            for t, records_in_trick in tricks.items():
                n = records_in_trick[0]["observation"]["num_players"]
                if len(records_in_trick) != n:
                    raise ValueError("Unfinished trick")
                winner_pos = trick_winner(
                    [CARD_ID[x["chosen_card"]] for x in records_in_trick],
                    SUITS.index(records_in_trick[0]["observation"]["trump"]))
                winner = records_in_trick[winner_pos]["player"]
                for rec in records_in_trick:
                    if (rec["behavior_policy"] != "BidAware3Player"
                            or len(rec["observation"]["own_hand"]) != 3
                            or len(rec["legal_cards"]) < 2):
                        continue
                    s = rec["observation"]
                    if s["pile"] != [[x["player"], x["chosen_card"]]
                                     for x in records_in_trick[:len(s["pile"])]]:
                        raise AssertionError("Public trick prefix disagrees")
                    selected.append({
                        "game": str(path.stem), "round": r, "trick": t,
                        "player": rec["player"],
                        "position": len(s["pile"]), "players": n,
                        "record": rec,
                        "won_trick": int(winner == rec["player"]),
                    })
    return selected, game_count


def reconstructed_state(item):
    s = InformationState.from_dict(item["record"]["observation"])
    masked = [0] * s.num_players
    masked[s.observer_id] = hand_mask(s.own_hand)
    # Minimal same interface as the real policy, NO opponent hands.
    from types import SimpleNamespace
    return s, SimpleNamespace(
        hands=tuple(masked), pile=[CARD_ID[c] for _,c in s.pile],
        turn=s.observer_id, bids=s.bids, tricks_won=s.tricks_won,
        trump=SUITS.index(s.trump))


def empirical_probability_rows(items):
    out = []
    for item in items:
        s, state = reconstructed_state(item)
        chosen = item["record"]["chosen_card"]
        p = _current_win_probability(state, CARD_ID[chosen], _THREE_CARD_WIN)
        out.append({
            "game": item["game"], "players": item["players"],
            "position": item["position"], "opponents_after":
                item["players"]-item["position"]-1,
            "bid_need": s.bids[s.observer_id]-s.tricks_won[s.observer_id],
            "p_predicted": p, "won": item["won_trick"],
            "brier": (p-item["won_trick"])**2,
        })
    return out


def bucket_probability(p):
    return min(4, int(p * 5))


def probability_summary(rows):
    if not rows:
        return {"n":0}
    pmean = statistics.mean(x["p_predicted"] for x in rows)
    actual = statistics.mean(x["won"] for x in rows)
    buckets=[]
    for i in range(5):
        subset=[r for r in rows if bucket_probability(r["p_predicted"])==i]
        if subset:
            buckets.append({
                "predicted_bin":f"{i/5:.1f}–{(i+1)/5:.1f}",
                "n":len(subset),
                "mean_predicted":statistics.mean(r["p_predicted"] for r in subset),
                "observed":statistics.mean(r["won"] for r in subset),
            })
    return {
        "n":len(rows), "mean_predicted":pmean, "observed_win_rate":actual,
        "prediction_bias":pmean-actual,
        "mean_brier":statistics.mean(r["brier"] for r in rows),
        "mean_absolute_calibration_gap":sum(
            b["n"]*abs(b["mean_predicted"]-b["observed"])
            for b in buckets)/len(rows),
        "bins":buckets,
    }


def all_probability_reports(rows):
    report={"overall":probability_summary(rows)}
    report["by_players"]={}
    report["by_position"]={}
    report["by_opponents_after"]={}
    for n in sorted({r["players"] for r in rows}):
        subset=[r for r in rows if r["players"]==n]
        report["by_players"][str(n)]=probability_summary(subset)
        report["by_position"][str(n)]={
            str(pos):probability_summary([
                r for r in subset if r["position"]==pos])
            for pos in range(n)}
        report["by_opponents_after"][str(n)]={
            str(k):probability_summary([
                r for r in subset if r["opponents_after"]==k])
            for k in range(n)}
    return report


def rank_and_regret(items, *, max_states_per_player=240, rollouts=1024,
                    seed=20261102):
    if rollouts < 1 or max_states_per_player < 1:
        raise ValueError("Need positive reference budget")
    rng=random.Random(seed)
    per_players=defaultdict(list)
    for item in items:
        per_players[item["players"]].append(item)
    out=[]
    for n, group in sorted(per_players.items()):
        # Sample reproducibly without looking at any outcomes or hand values.
        chosen = rng.sample(group, min(max_states_per_player,len(group)))
        for j,item in enumerate(chosen):
            obs, state = reconstructed_state(item)
            legal = tuple(item["record"]["legal_cards"])
            heuristic=three_card_values(state,hand_mask(legal))
            if heuristic is None:
                raise AssertionError("Expected a three-card choice")
            cards, vals=heuristic
            heuristic_q=dict(zip(cards,vals))
            hbest = max(cards, key=lambda card:heuristic_q[card])
            ref_seed = seed + n*1000000007 + j*100003
            ref=evaluate_actions(
                obs,legal,n_rollouts=rollouts,seed=ref_seed,workers=1,
                policy="bid_aware_2")
            mc_q={CARD_ID[action]:v for action,v in zip(ref.actions,ref.values)}
            mc_best=max(cards,key=lambda card:mc_q[card])
            mc_regret=mc_q[mc_best]-mc_q[hbest]
            mc_gap=sorted((mc_q[x] for x in cards),reverse=True)
            decisiveness=mc_gap[0]-mc_gap[1]
            out.append({
                "game":item["game"],"players":n,"round":item["round"]+1,
                "trick":item["trick"]+1,"player":item["player"],
                "position":item["position"],
                "bid":obs.bids[obs.observer_id],
                "won_before":obs.tricks_won[obs.observer_id],
                "trump":obs.trump,"hand":list(obs.own_hand),
                "pile":list(obs.pile),"actual_chosen":item["record"]["chosen_card"],
                "heuristic_best":CARDS[hbest],"reference_best":CARDS[mc_best],
                "heuristic_q":{CARDS[c]:round(heuristic_q[c],3) for c in cards},
                "reference_q":{CARDS[c]:round(mc_q[c],3) for c in cards},
                "heuristic_correct":int(hbest==mc_best),
                "regret":mc_regret,
                "reference_best_second_gap":decisiveness,
                "reference_seed":ref_seed,
                "observation":item["record"]["observation"],
                "legal_cards":list(legal),
            })
            if (j+1)%60==0:
                print(f"AUDIT players={n} complete={j+1}/{len(chosen)}",flush=True)
    return out


def ranking_summary(rows):
    if not rows:return {"n":0}
    return {
        "n":len(rows),
        "action_agreement":statistics.mean(r["heuristic_correct"] for r in rows),
        "mean_reference_regret":statistics.mean(r["regret"] for r in rows),
        "median_reference_regret":statistics.median(r["regret"] for r in rows),
        "fraction_regret_gt_1":statistics.mean(r["regret"]>1 for r in rows),
        "fraction_regret_gt_2":statistics.mean(r["regret"]>2 for r in rows),
        "fraction_decisive_mc_gt_1":statistics.mean(
            r["reference_best_second_gap"]>1 for r in rows),
        "agreement_given_decisive_mc":(
            statistics.mean(r["heuristic_correct"] for r in rows
                            if r["reference_best_second_gap"]>1)
            if any(r["reference_best_second_gap"]>1 for r in rows) else None),
    }


def all_ranking_reports(rows):
    return {
        str(n): {
            "overall":ranking_summary([r for r in rows if r["players"]==n]),
            "by_position":{
                str(p):ranking_summary([r for r in rows
                                        if r["players"]==n and r["position"]==p])
                for p in range(n)},
            "by_bid_need":{
                category:ranking_summary([
                    r for r in rows if r["players"]==n and (
                        (r["bid"]-r["won_before"]<=0) if category=="met_or_over"
                        else (r["bid"]-r["won_before"]==1 if category=="need_one"
                              else r["bid"]-r["won_before"]>=2))])
                for category in ("met_or_over","need_one","need_two_plus")},
        }
        for n in sorted({r["players"] for r in rows})
    }


def confirm_disagreements(rows, *, top_per_player=12, rollouts=8192, seed=20261102):
    """Independent reroll of largest estimated action differences.

    **Selection bias**: selected for high initial regret; report separately
    rather than combining into the original summary.
    """
    confirmed=[]
    for n in sorted({r["players"] for r in rows}):
        subset=sorted((r for r in rows if r["players"]==n),
                      key=lambda r:r["regret"],reverse=True)[:top_per_player]
        for i,r in enumerate(subset):
            state=InformationState.from_dict(r["observation"])
            result=evaluate_actions(
                state,r["legal_cards"],n_rollouts=rollouts,
                seed=seed+999999999*n+i*271828,policy="bid_aware_2",
                workers=1)
            q=dict(zip(result.actions,result.values))
            refbest=max(q,key=q.get)
            r2={k:v for k,v in r.items() if k not in ("observation",)}
            r2.update({
                "fresh_best":refbest,
                "fresh_regret":q[refbest]-q[r["heuristic_best"]],
                "fresh_score_values":{k:round(v,3) for k,v in q.items()},
            })
            confirmed.append(r2)
        print(f"CONFIRM players={n} evaluated={len(subset)}",flush=True)
    return confirmed


def markdown_report(probability, rankings, confirmed, meta):
    out=[
        "# Bid-aware three-card diagnostic: actual outcomes and reference MC",
        "",
        f"Recorded actual complete games: {meta['games']}",
        f"3-card nonforced chosen plays audited for empirical calibration:"
        f" {meta['calibration_decisions']}",
        f"MC-ranked decision states: {meta['mc_states']} across 2P and 3P",
        "",
        "## 1. Does the current-trick win probability match real outcomes?",
        "",
        "| Players | Acting position | N | Predicted win | Actual win | Bias | Brier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for n,pos_map in probability["by_position"].items():
        for pos, s in pos_map.items():
            if s["n"]:
                out.append(
                    f"| {n} | {int(pos)+1} of {n} | {s['n']} | "
                    f"{s['mean_predicted']:.1%} | "
                    f"{s['observed_win_rate']:.1%} | "
                    f"{s['prediction_bias']:+.1%} | {s['mean_brier']:.3f} |")
    out.extend([
        "",
        "For calibration, outcomes are the ACTUAL current-trick winner for "
        "the card the policy played, not simulated labels. "
        "A prediction of p(win) is evaluated only on the actual action.",
        "The empirical calibration sample reflects the policy's choice "
        "distribution and does not cover counterfactual cards.",
        "",
        "## 2. Does the action ranking agree with independent MC?",
        "",
        "| Players | Position | N | Best-card agreement | Mean MC regret "
        "| % regrets > 1pt |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for n, s in rankings.items():
        for pos, value in s["by_position"].items():
            if value["n"]:
                out.append(
                    f"| {n} | {int(pos)+1} of {n} | {value['n']} | "
                    f"{value['action_agreement']:.1%} | "
                    f"{value['mean_reference_regret']:.2f} | "
                    f"{value['fraction_regret_gt_1']:.1%} |")
    out.extend([
        "",
        "Reference Monte Carlo samples ONLY hands consistent with the acting "
        "player's visible information, with shared hidden deals and random "
        "seeds across candidate cards. It uses bid_aware_2 continuation for "
        "all simulated players; it is not perfect ground truth, nor an "
        "estimate of full-game winning probability.",
        "",
        "## 3. Independently rerun biggest disagreements",
        "",
        "| Players | Position | Round | Hand | Chosen by heuristic "
        "| Initial regret | Fresh independent regret | Fresh MC best |",
        "|---|---|---:|---|---|---:|---:|---|",
    ])
    for item in sorted(confirmed,key=lambda r:r["fresh_regret"],reverse=True)[:20]:
        out.append(
            f"| {item['players']} | {item['position']+1} | {item['round']} | "
            f"{','.join(item['hand'])} | {item['heuristic_best']} | "
            f"{item['regret']:.2f} | {item['fresh_regret']:.2f} | "
            f"{item['fresh_best']} |")
    out.extend([
        "",
        "## Important limitations",
        "",
        "- Estimated regret uses a finite-rollout, hand-agnostic continuation "
        "reference; even fresh confirmation has variance and potential "
        "continuation-model bias.",
        "- Win-probability calibration on *chosen plays* does not establish "
        "how well each unchosen card's probabilities are calibrated.",
        "- More players also dilute individual control of a trick; poor "
        "ranking and low control can coexist. The diagnostic helps "
        "distinguish them but does not prove causality.",
        "- Player bidding is still the original RandomPlayer average bid, "
        "not MC bidding; both populations share that constraint.",
    ])
    return "\n".join(out)+"\n"


def run(*, output_dir, games_per_players=300, max_states_per_player=240,
        rollouts=1024, confirm_rollouts=8192, confirm_top=12,
        seed=20261131, workers=4, reuse_traces=False):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    all_items=[]
    counts={}
    for n in (2,3):
        directory=output_dir/f"traces-{n}p"
        if not reuse_traces:
            generate(games=games_per_players,seed=seed+100000*n,
                     players=n,agents=("aware3","random"),
                     workers=workers,output_dir=directory)
        items,count=read_decisions(directory)
        all_items+=items
        counts[str(n)]={"games":count,"states":len(items)}
        print(f"TRACES {n}P games={count} three_card_decisions={len(items)}",
              flush=True)
    probabilities=all_probability_reports(empirical_probability_rows(all_items))
    print("PROBABILITIES "+json.dumps({
        n:{"n":v["n"],"bias":v["prediction_bias"],
           "brier":v["mean_brier"]}
        for n,v in probabilities["by_players"].items()}),flush=True)
    ranks=rank_and_regret(
        all_items,max_states_per_player=max_states_per_player,
        rollouts=rollouts,seed=seed+100)
    ranking_reports=all_ranking_reports(ranks)
    print("RANKING "+json.dumps({
        n:s["overall"] for n,s in ranking_reports.items()}),flush=True)
    confirmed=confirm_disagreements(
        ranks,top_per_player=confirm_top,
        rollouts=confirm_rollouts,seed=seed+200)
    metadata={"games":counts,"calibration_decisions":len(all_items),
              "mc_states":len(ranks),"mc_rollouts":rollouts,
              "confirm_rollouts":confirm_rollouts,
              "seed":seed,"workers":workers}
    result={"metadata":metadata,"probability_calibration":probabilities,
            "mc_ranking":ranking_reports,"confirmed":confirmed}
    (output_dir/"summary.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (output_dir/"decision-comparisons.jsonl").open("w",encoding="utf-8") as f:
        for row in ranks:
            f.write(json.dumps({
                k:v for k,v in row.items() if k!="observation"})+"\n")
    report=markdown_report(probabilities,ranking_reports,confirmed,metadata)
    (output_dir/"analysis.md").write_text(report,encoding="utf-8")
    print("ANALYSIS_BEGIN\n"+report+"ANALYSIS_END",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--games-per-players",type=int,default=300)
    p.add_argument("--max-states-per-player",type=int,default=240)
    p.add_argument("--rollouts",type=int,default=1024)
    p.add_argument("--confirm-rollouts",type=int,default=8192)
    p.add_argument("--confirm-top",type=int,default=12)
    p.add_argument("--seed",type=int,default=20261131)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--reuse-traces",action="store_true")
    a=p.parse_args()
    run(output_dir=a.output_dir,games_per_players=a.games_per_players,
        max_states_per_player=a.max_states_per_player,
        rollouts=a.rollouts,confirm_rollouts=a.confirm_rollouts,
        confirm_top=a.confirm_top,seed=a.seed,workers=a.workers,
        reuse_traces=a.reuse_traces)


if __name__=="__main__":
    main()
