"""Paired diagnostic comparing original and analytical 3-card Q estimates.

Uses actor-only recorded game states; no true opponent hands. Shares
the SAME MC reference candidate Q values across both compared heuristics.
Reports absolute improvement or deterioration, not just action agreement.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from alphawhist.core.AnalyticEndgame import (
    analytical_three_card_values, current_trick_win_probability,
    two_card_joint_value,
)
from alphawhist.core.EndgamePolicy import (
    three_card_values, _TEMPERATURE, _two_card_continuation_value,
    _THREE_CARD_WIN, _current_win_probability,
)
from alphawhist.core.FastRollout import SUITS
from alphawhist.core.FastRollout import CARD_ID, CARDS, evaluate_actions, hand_mask
from benchmarks.generate_training_games import generate
from benchmarks.three_card_calibration import read_decisions, reconstructed_state


def _softmax(scores):
    from math import exp
    mx=max(scores.values())
    weights={k:exp((v-mx)/_TEMPERATURE) for k,v in scores.items()}
    scale=sum(weights.values())
    return {k:v/scale for k,v in weights.items()}


def compare(states, *, count=120, mc_rollouts=1024, seed=20261102):
    rng=random.Random(seed)
    rows=[]
    by_player=defaultdict(list)
    for item in states:
        by_player[item["players"]].append(item)
    for players,group in sorted(by_player.items()):
        chosen=rng.sample(group,min(count,len(group)))
        for idx,item in enumerate(chosen):
            obs,masked=reconstructed_state(item)
            actions=tuple(item["record"]["legal_cards"])
            before=three_card_values(masked,hand_mask(actions))
            old={CARDS[c]:v for c,v in zip(*before)}
            new=dict(analytical_three_card_values(obs,actions))
            if set(new)!=set(old):
                raise AssertionError("Analytical and old actions differ")
            # Ablate the two changes independently on identical observations:
            # HG-only = combinatorial P(win now) + old rank continuation.
            # Joint-only = old P(win now) + analytic joint two-trick continuation.
            hypergeom_only={}
            joint_only={}
            me=obs.observer_id
            needed=obs.bids[me]-obs.tricks_won[me]
            nopp=obs.num_players-1
            trump=SUITS.index(obs.trump)
            pools=tuple(tuple(sorted(CARD_ID[x] for x in opp.possible_cards))
                        for opp in obs.opponents)
            for action in actions:
                remaining=[CARD_ID[x] for x in obs.own_hand if x!=action]
                a,b=remaining
                old_win=_two_card_continuation_value(
                    a,b,needed-1,True,trump,nopp)
                old_loss=_two_card_continuation_value(
                    a,b,needed,False,trump,nopp)
                joint_win=two_card_joint_value(
                    a,b,needed-1,True,pools,trump)
                joint_loss=two_card_joint_value(
                    a,b,needed,False,pools,trump)
                p_hg=current_trick_win_probability(obs,CARD_ID[action])
                p_old=_current_win_probability(
                    masked,CARD_ID[action],_THREE_CARD_WIN)
                hypergeom_only[action]=p_hg*(1+old_win)+(1-p_hg)*old_loss
                joint_only[action]=p_old*(1+joint_win)+(1-p_old)*joint_loss
            reference=evaluate_actions(
                obs,actions,n_rollouts=mc_rollouts,
                seed=seed+players*1000003+idx*16381,
                policy="bid_aware_2",workers=1)
            q=dict(zip(reference.actions,reference.values))
            best=max(q,key=q.get)
            old_best=max(old,key=old.get)
            new_best=max(new,key=new.get)
            old_policy=_softmax(old)
            new_policy=_softmax(new)
            hg_policy=_softmax(hypergeom_only)
            joint_policy=_softmax(joint_only)
            old_regret=q[best]-sum(old_policy[a]*q[a] for a in actions)
            new_regret=q[best]-sum(new_policy[a]*q[a] for a in actions)
            hg_regret=q[best]-sum(hg_policy[a]*q[a] for a in actions)
            joint_regret=q[best]-sum(joint_policy[a]*q[a] for a in actions)
            rows.append({
                "game":item["game"],"players":players,
                "position":item["position"],
                "round":item["round"]+1,
                "hand":list(obs.own_hand),"pile":list(obs.pile),
                "bid":obs.bids[obs.observer_id],
                "won_before":obs.tricks_won[obs.observer_id],
                "actions":list(actions),
                "old_best":old_best,"new_best":new_best,
                "hypergeom_only_best":max(hypergeom_only,key=hypergeom_only.get),
                "joint_only_best":max(joint_only,key=joint_only.get),
                "mc_best":best,
                "old_best_agrees":int(old_best==best),
                "new_best_agrees":int(new_best==best),
                "old_regret":old_regret,
                "new_regret":new_regret,
                "hypergeom_only_regret":hg_regret,
                "joint_only_regret":joint_regret,
                "regret_improvement":old_regret-new_regret,
                "old_q":old,"new_q":new,
                "hypergeom_only_q":hypergeom_only,
                "joint_only_q":joint_only,"mc_q":q,
            })
        print(f"MC_AUDIT players={players} n={len(chosen)}",flush=True)
    return rows


def summarize(rows):
    result={}
    for n in sorted({x["players"] for x in rows}):
        subset=[x for x in rows if x["players"]==n]
        result[str(n)]={
            "n":len(subset),
            "old_best_agreement":statistics.mean(x["old_best_agrees"] for x in subset),
            "analytic_best_agreement":statistics.mean(x["new_best_agrees"] for x in subset),
            "old_softmax_regret":statistics.mean(x["old_regret"] for x in subset),
            "analytic_softmax_regret":statistics.mean(x["new_regret"] for x in subset),
            "hypergeom_only_softmax_regret":statistics.mean(
                x["hypergeom_only_regret"] for x in subset),
            "joint_only_softmax_regret":statistics.mean(
                x["joint_only_regret"] for x in subset),
            "hypergeom_only_best_agreement":statistics.mean(
                x["hypergeom_only_best"]==x["mc_best"] for x in subset),
            "joint_only_best_agreement":statistics.mean(
                x["joint_only_best"]==x["mc_best"] for x in subset),
            "mean_regret_improvement":statistics.mean(x["regret_improvement"] for x in subset),
            "fraction_analytic_better":statistics.mean(x["regret_improvement"]>0
                                                      for x in subset),
            "by_position":{
                str(pos): {
                    "n":sum(x["position"]==pos for x in subset),
                    "mean_regret_improvement":statistics.mean(
                        x["regret_improvement"] for x in subset if x["position"]==pos)
                    if any(x["position"]==pos for x in subset) else None}
                for pos in range(n)},
        }
    return result


def empirical_calibration(items):
    result={}
    for n in sorted({x["players"] for x in items}):
        subset=[x for x in items if x["players"]==n]
        def score(rows):
            if not rows:return None
            errors=[]
            old_errors=[]
            predictions=[]
            outcomes=[]
            from alphawhist.core.EndgamePolicy import (
                _current_win_probability,_THREE_CARD_WIN)
            for item in rows:
                snap,masked=reconstructed_state(item)
                card=CARD_ID[item["record"]["chosen_card"]]
                p=current_trick_win_probability(snap,card)
                old=_current_win_probability(masked,card,_THREE_CARD_WIN)
                winner=item["won_trick"]
                errors.append((p-winner)**2)
                old_errors.append((old-winner)**2)
                predictions.append(p)
                outcomes.append(winner)
            return {
                "n":len(rows),"new_brier":statistics.mean(errors),
                "old_brier":statistics.mean(old_errors),
                "new_predicted":statistics.mean(predictions),
                "empirical_win_rate":statistics.mean(outcomes),
                "brier_improvement":statistics.mean(old_errors)-statistics.mean(errors),
            }
        result[str(n)]={
            "overall":score(subset),
            "by_position":{str(i):score(
                [x for x in subset if x["position"]==i]) for i in range(n)}
        }
    return result


def run(*, output_dir, games_per_player=80, max_states=120,
        mc_rollouts=1024, workers=4, seed=20261102):
    path=Path(output_dir)
    path.mkdir(parents=True,exist_ok=True)
    items=[]
    for n in (2,3):
        traces=path/f"traces-{n}p"
        generate(games=games_per_player,players=n,
                 agents=("aware3","random"),workers=workers,
                 seed=seed+n*10000,output_dir=traces)
        states,games=read_decisions(traces)
        if games!=games_per_player:
            raise AssertionError("Incomplete evaluation traces")
        items+=states
    cal=empirical_calibration(items)
    print("CALIBRATION "+json.dumps(cal,sort_keys=True),flush=True)
    decisions=compare(items,count=max_states,mc_rollouts=mc_rollouts,seed=seed)
    ranked=summarize(decisions)
    print("RANKING "+json.dumps(ranked,sort_keys=True),flush=True)
    result={
        "metadata":{"games_each":games_per_player,"reference_rollouts":mc_rollouts,
                    "max_states_per_player":max_states,"seed":seed},
        "calibration":cal,"ranking":ranked}
    (path/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    with (path/"decisions.jsonl").open("w") as f:
        for row in decisions:
            f.write(json.dumps(row)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--games-per-player",type=int,default=80)
    p.add_argument("--max-states",type=int,default=120)
    p.add_argument("--mc-rollouts",type=int,default=1024)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261102)
    a=p.parse_args()
    run(output_dir=a.output_dir,games_per_player=a.games_per_player,
        max_states=a.max_states,mc_rollouts=a.mc_rollouts,
        workers=a.workers,seed=a.seed)


if __name__=="__main__":
    main()
