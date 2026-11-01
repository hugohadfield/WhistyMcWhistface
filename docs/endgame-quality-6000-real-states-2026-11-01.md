# Bid-aware two-card endgame: 6,000 realistic decision states

**Date:** 1 November 2026  
**Repository branch:** `experiment/endgame-realistic-quality-2to7`  
**Source:** [endgame quality evaluator](../benchmarks/endgame_quality.py), built on the actual game engine and [FastRollout](../alphawhist/core/FastRollout.py)  
**Experiment CI:** [successful six-job matrix run #41334554851](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851)  
**Reference implementation:** `finish_round(..., "random")` with independently sampled hidden worlds for each observed information state. **This is not a perfect-play oracle.**

## Summary

The earlier quality diagnostic tested only **50 artificial, complete-hand states per player count** and estimated the value of each action using **96 simulations**. It found just 46–60% best-action agreement. Those artificial states had randomly fabricated bid and trick-count histories, and the reference already knew the *true* opponent hands, so the test did not reflect how the policy is actually used.

This follow-up extracts **1,000 two-card information states per player count from independent, legally played rounds** under the authoritative Whist game engine: **6,000 independent decision states** across 2–7 players. It uses **2,048 sampled hidden worlds** per information state, common across both candidate moves, for **12,288,000 hidden-hand determinizations** and **24,576,000 candidate-action continuations**. Every reference continuation follows legal card-play rules and uses independent seeded random opponent actions.

### Main outcome

| Players | Independent states | Greedy heuristic agreement | Agreement on resolved decisions* | Greedy regret (points) | Stochastic regret (points) | Random regret (points) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1,000 | **90.8%** | 95.55% (876) | 0.0242 | 0.3070 | 0.7215 |
| 3 | 1,000 | **88.4%** | 93.76% (849) | 0.0365 | 0.2940 | 0.6389 |
| 4 | 1,000 | **84.8%** | 87.63% (865) | 0.1046 | 0.3141 | 0.5554 |
| 5 | 1,000 | **83.8%** | 87.77% (842) | 0.1408 | 0.3280 | 0.5422 |
| 6 | 1,000 | **81.7%** | 86.86% (814) | 0.1212 | 0.3029 | 0.5215 |
| 7 | 1,000 | **78.1%** | 85.06% (716) | 0.1653 | 0.3045 | 0.4765 |
| **Equal-weight average** | **6,000** | **84.6%** | **~89.6% (4,962)** | **0.0988** | **0.3084** | **0.5760** |

*Resolved means that the paired Monte Carlo action-value difference exceeds 1.96 times its estimated paired Monte Carlo standard error. This is a descriptive noise filter, not ground-truth optimality or a multiple-testing-adjusted significance test.*

The heuristic is **substantially better than the original 46–60% numbers implied** when evaluated on legitimate information states. It appears to approximate the *random-continuation value reference* quite well in 2–4 player games and less well, but still substantially better than chance, in 5–7 player games.

The *production opt-in policy* samples a softmax with temperature 1.5 rather than choosing the greedy card every time. Its mean regret of **0.308** is ~3× the greedy heuristic's **0.099** points. Its behavior appears unnecessarily stochastic relative to maximizing the current Monte Carlo reference value; **a temperature ablation is worthwhile** before enabling it by default.

### Actual improvement over choosing a uniformly random card

The greedy, stochastic and trivial bid-aware high/low rules are evaluated against **exactly the same paired reference action scores**, so differences in score regret are directly comparable at each state.

| Players | Greedy bid-aware gain vs random | Stochastic bid-aware gain vs random | Simple bid-aware high/low gain vs random |
| ---: | ---: | ---: | ---: |
| 2 | **+0.6973** | **+0.4145** | +0.4039 |
| 3 | **+0.6024** | **+0.3448** | +0.2492 |
| 4 | **+0.4508** | **+0.2413** | +0.0978 |
| 5 | **+0.4014** | **+0.2142** | +0.0636 |
| 6 | **+0.4003** | **+0.2186** | +0.0442 |
| 7 | **+0.3112** | **+0.1720** | +0.0229 |
| **Equal-weight average** | **+0.4772** | **+0.2676** | +0.1469 |

All six stochastic gain 95% confidence intervals are positive. The 7-player high/low baseline's interval **includes zero**: +0.0229 [−0.0342, +0.0800]. This makes the difference between the probabilistic heuristic and a hard-coded bid-aware high/low strategy meaningful.

### Per-player-count uncertainty

Means and approximate **95% normal intervals** are computed over the 1,000 independent game-round seeds at each player count. These reflect variability across decision states; they do not fully incorporate Monte Carlo estimation error in the reference nor prove that a policy is strategically stronger against non-random opponents.

| Players | Greedy regret, 95% CI | Stochastic regret, 95% CI | Random regret, 95% CI | Stochastic gain vs random, 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0242 [0.0141, 0.0343] | 0.3070 [0.2893, 0.3246] | 0.7215 [0.6669, 0.7761] | +0.4145 [+0.3636, +0.4654] |
| 3 | 0.0365 [0.0225, 0.0505] | 0.2940 [0.2771, 0.3109] | 0.6389 [0.5944, 0.6833] | +0.3448 [+0.3072, +0.3825] |
| 4 | 0.1046 [0.0770, 0.1323] | 0.3141 [0.2924, 0.3357] | 0.5554 [0.5125, 0.5983] | +0.2413 [+0.2064, +0.2763] |
| 5 | 0.1408 [0.1059, 0.1758] | 0.3280 [0.3022, 0.3539] | 0.5422 [0.4966, 0.5879] | +0.2142 [+0.1771, +0.2514] |
| 6 | 0.1212 [0.0881, 0.1543] | 0.3029 [0.2755, 0.3303] | 0.5215 [0.4714, 0.5716] | +0.2186 [+0.1773, +0.2598] |
| 7 | 0.1653 [0.1250, 0.2056] | 0.3045 [0.2734, 0.3356] | 0.4765 [0.4276, 0.5254] | +0.1720 [+0.1351, +0.2090] |

## How states were collected (important distinction)

Each test case uses a **fresh random seed** and a **complete, legally played game round** with 2–7 players and 2–7 cards per player. The round lengths are cycled so all lengths have representation at each player count.

The game engine uses its existing RandomPlayer card-play, trump, and bidding methods; bids are **legal real bids produced by the game's rules and its RandomPlayer bidding policy**, *not fabricated independent random integers*. We record all public plays, known void suits, lead/trump, bids and tricks won, and capture a public `InformationState` for every player facing two legal card choices with two cards remaining. At the end of each round, **one** of these snapshots is selected using an independent deterministic RNG. No two retained states share the same underlying round seed.

**Scope caveat:** These are complete **rounds** sampled independently, not whole 13-round matches. Accordingly, they do not cover prior-round consecutive-zero-bid constraints or strategic game-score management. That is acceptable for comparing the current 2-card round-score model, but it is not a full-match playing-strength evaluation. Also, because RandomPlayer uses simple bidding and random trick play, these are **legal** rather than representative of expert Whist states.

For each snapshot, the acting player's two cards and public state feed the heuristic. Its decisions are computed **without the true opponent hands**. The reference then draws independent opponent deals from `InformationState.sample_hands`, which respects public card/void constraints and known unavailable cards. It simulates each candidate's remainder of the round using exactly the **same sampled hidden world and continuation RNG seed**; this substantially reduces comparison variance.

The resulting two action-score estimates `Q_a`, `Q_b` give:

\[
\mathrm{regret}(\pi\mid I)=\max(Q_a,Q_b)-\mathbb E_{a\sim\pi(\cdot\mid I)}[Q_a].
\]

We report empirical means and confidence intervals over independent states. This is **reference-model regret**, not true strategy regret. In addition to the current stochastic and greedy `bid_aware_2` heuristic, we evaluate a simple deterministic high/low rule keyed to bid deficit and a uniform random action rule.

## Experiment details

- Player counts: 2, 3, 4, 5, 6, 7; 6 independent CI jobs
- Unique legal states: 1,000 per player count, **6,000 total**
- Hidden worlds per state: **2,048** independent samples drawn from public information
- Continuations: two per hidden world, one for each legal card, using common random numbers
- Total: **12,288,000 hidden worlds; 24,576,000 candidate continuations**
- Random seed: `20261101`, mixed deterministically with player count and state index
- Observational data: one independent complete round per retained state
- Temperature: `1.5` (existing policy, not tuned on these results)
- Reference future play: **uniform random** for all remaining non-forced moves, *including opponents*
- 95% intervals: normal ±1.96 standard errors over 1,000 distinct decisions, per group
- Python regression suite before launching the matrix: **487 passed**
- CI durations for reference stages by player count: 2p 94.25 s; 3p 66.93 s; 4p 92.64 s; 5p 134.20 s; 6p 124.38 s; 7p 248.31 s. Each ran on its own GitHub Actions runner. These are not per-move production latencies.

## Artifacts

The successful [six-job CI run](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851) contains one CSV artifact for each player count. GitHub retains them for **14 days**. The report and deterministic generator code remain in git.

- [2-player results](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851/artifacts/13094965908)
- [3-player results](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851/artifacts/13094681384)
- [4-player results](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851/artifacts/13095191074)
- [5-player results](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851/artifacts/13094611621)
- [6-player results](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851/artifacts/13095175572)
- [7-player results](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41334554851/artifacts/13094917220)

Each CSV contains the legal initial round size, observer position, public trick status, actual legal bid, current tricks won, two candidate cards, bid-aware heuristic values, independently sampled Monte Carlo reference scores, estimated paired-reference difference standard error, reference ambiguity, simple baseline actions and per-policy regret.

### Reproduce

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_endgame_quality.py tests/test_endgame_policy.py
python -m benchmarks.endgame_quality \
  --players 3 --states 1000 --rollouts 2048 --seed 20261101 \
  --csv endgame-quality-3.csv
```

Change `--players` from 2 to 7; run in six parallel processes / CI jobs for throughput. The original `bid_aware_2` agent remains **opt-in**.

## What to do next

These data strongly support the usefulness of **bid-aware two-card action ordering under random-continuation scoring**. However, they do not validate that the **stochastic action policy is a realistic model of stronger opponents**, and they do not demonstrate higher game win rate. More realistic next experiments would be:

1. A temperature ablation on held-out independent states (`tau` near 0, 0.25, 0.5, 1, 1.5, 3), comparing expected-score regret and predictive fit to empirical strategic players.
2. Paired 200–500 game tournaments with the same rollouts and bid/trump selectors, changing only `monteCarloPolicy`, against both random and stronger opponents.
3. A learned or calibrated opponent action likelihood using actual bids and public card history, not just a low/high card prior, to remove model mismatch at 5–7 players.
4. Separate held-out references under **alternative future-play policies**, to establish whether the heuristic remains good when opponents don't play randomly.

**Conclusion:** The earlier 46–60% "agreement" suggested weak quality, but the larger, more relevant hidden-information experiment changes that assessment. The greedy heuristic ranks actions well, while the production stochastic policy is more conservative. Claims about match-playing strength must still wait for competitive tournaments.
