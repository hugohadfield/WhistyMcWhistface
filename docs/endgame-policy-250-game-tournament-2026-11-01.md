# Bid-aware versus random future play: 250-game head-to-head tournament

**Date:** 1 November 2026  
**Game:** 3-player contract Whist, 13 rounds per match (55 tricks, 165 card plays)  
**Experiment:** [successful GitHub Actions run #41338768757](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41338768757)  
**Permanent per-match data:** [250 match CSV](../benchmarks/results/endgame-policy-250-2026-11-01.csv)  
**Temporary original CSV artifact:** [GitHub Actions artifact #13096703165](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41338768757/artifacts/13096703165), expires 15 November 2026.  
**Comparison code:** [`endgame_policy_tournament.py`](../benchmarks/endgame_policy_tournament.py)

## Result: modest positive trend, uncertainty crosses zero

| Quantity | Bid-aware two-card continuation | Uniform random continuation |
| --- | ---: | ---: |
| **Head-to-head games won** | **134 / 250** | **114 / 250** |
| Head-to-head ties | 2 | 2 |
| Outright 3-player first-place finishes | **132 / 250** | **109 / 250** |
| Mean final game points | **94.404** | **93.216** |
| Median head-to-head margin (aware − random) | **+2 points** | — |

The third, non-Monte-Carlo `RandomPlayer` outright won **6** matches. Another **3** matches tied for first. Thus the two competitors won 241 games outright (132 + 109) and the remaining 9 belong to the random competitor or ties.

**Score difference:** **+1.188 final game points** in favor of the bid-aware continuation policy.

**Paired bootstrap 95% interval:** **[−0.152, +2.472] points per game**.

The interval includes zero: 250 games do **not** conclusively show higher overall playing strength. The positive trend is consistent across raw head-to-head wins and average score, but could plausibly arise from game/deal variance. It would be inappropriate to enable the policy by default based solely on this experiment.

### Paired-seed analysis

Each of the **125 independent deal seeds** is played twice, swapping the competing agents' seats. At the pair level (averaging each pair's two head-to-head score margins):

- **58 pairs** favored the bid-aware policy.
- **50 pairs** favored random-continuation policy.
- **17 pairs** produced an exactly tied average.
- The paired mean was **+1.188 points/game**.
- 20,000-resample percentile bootstrap, resampling *complete two-game pairs* with replacement, yielded **[−0.152,+2.472]**.

The pair is the sampling unit for uncertainty. Treating the 250 games as 250 independent draws would incorrectly neglect shared initial deals between each pair's games.

## Controlled experimental design

Both competing players are instances of the same `MonteCarloPlayer` with:

- Identical `best_trump_by_expected_score` method for choosing trumps.
- Identical `makeAdvancedBid` algorithm for bidding.
- Identical immediate card-selection objective: maximizing expected own-round points.
- Identical **1,024 hidden-world rollouts per non-forced card decision** and a single rollout worker per player.
- **Only intentional difference:** the **continuation** policy used *for all players inside each simulated future*: `policy="bid_aware_2"` versus `policy="random"`.

A third `RandomPlayer` completes the three-player game. Starting hands for each of the **13 rounds** are regenerated using the same fixed `deal_seed*1009+roundnumber` before each round in both members of a seat-swap pair. The competitor seats cycle through (0,1), (0,2) and (1,2). This balances seat effects while retaining a random baseline opponent.

**Seed range:** `20261301`–`20261425`, 125 deal seeds, 250 games.  
**Execution:** four concurrent whole-game processes; each agent still uses one rollout worker.  
**Reproduce:**

```bash
python -m benchmarks.endgame_policy_tournament \
  --games 250 --rollouts 1024 --workers 4 \
  --seed 20261301 --csv endgame-policy-250.csv
```

**Why identical deals are not identical outcomes:** as the two algorithms play differently, later public states, random continuation seeds, trick wins, zero-bid counters and sometimes subsequent choices can diverge. Deal pairing controls starting hands and seats, not every stochastic action. Nor is the `RandomPlayer` guaranteed to take identical actions in paired games.

## Runtime

The 250 games took **839.773 seconds** (**13 minutes 59.8 seconds**) of tournament wall time on GitHub-hosted Actions with four whole-game workers. The sum of per-match elapsed times was **3,340.658 seconds**. These are not directly equivalent to a serial full-game runtime benchmark and do not isolate the extra CPU time of either policy.

The independent [endgame throughput experiment](bid-aware-endgame-policy-2026-11-01.md) found `bid_aware_2` adds about **25–28%** to a representative three-player seven-card complete decision evaluation, while isolated two-trick simulations were about **1.5–1.6×** random continuation. Those measurements remain relevant when evaluating whether the **+1.188 point** trend is worth the extra compute.

All **496 pytest tests passed** during the successful tournament CI run, including seat-swap and game-worker reproducibility regressions.

## Interpretation relative to the 6,000-state endgame experiment

The [6,000-state experiment](endgame-quality-6000-real-states-2026-11-01.md) found a greedy bid-aware move preference agreeing **84.6%** with a high-budget, random-continuation reference and reducing estimated action regret.

This tournament is the first end-to-end validation where the Monte Carlo action evaluator uses the two-card policy for *all simulated agents*, including both opponents and our own later simulated play. It changes how the agent **predicts future behavior**, not how it directly executes an independent two-card heuristic in live play. Immediate actions remain chosen by `evaluate_actions` based on 1,024 sampled outcomes.

The positive score and win-rate trend supports further investigation but cannot yet establish that the modeled future play is realistic, or that the policy would improve performance across 2–7 player games or against stronger opponents. The head-to-head comparison here is specifically **3-player play with a random third opponent**.

## Follow-up

1. **Run a temperature ablation**, e.g. deterministic greed versus `tau=0.25,0.5,1,1.5,3`, ideally evaluating same-deal tournaments or held-out sample states. The existing softmax temperature of 1.5 was not tuned.
2. **Evaluate 2–7 players** in tournaments or legal state simulations, rather than extrapolating the current three-player result.
3. **Test against strategic opponents**, because the reference estimator still uses a coarse rank/trump prior and ignores opponent bidding as evidence about hidden hands.
4. If the improvement remains small, compare **strength at equal CPU budgets**, not just equal rollout counts. The latter intentionally spends more computation on bid-aware continuation.

**Conclusion:** Bid-aware future-play simulation beat random future play **134–114** in this 250-game test and scored **+1.188 points per game on average**, but its 95% paired interval **crossed zero**. Leave random continuation the default pending stronger evidence.
