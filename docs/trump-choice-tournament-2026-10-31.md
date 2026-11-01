# Trump selection: exact-DP expected-score versus legacy heuristic — 50-game tournament

**Date:** 31 October 2026 Pacific / 1 November 2026 UTC  
**Implementation:** [PR #17](https://github.com/hugohadfield/WhistyMcWhistface/pull/17)  
**Reproducible runner:** [`benchmarks/trump_tournament.py`](../benchmarks/trump_tournament.py)  
**Original experiment and all scores:** [CI run #41283551004](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41283551004)  
**Temporary machine-readable CSV:** [50-game CSV artifact](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41283551004/artifacts/13073558910) (GitHub retention 14 days; full results preserved below).  
**Verification:** 430 regression tests passed, including new selector rules and head-to-head tournament smoke tests.

## Executive findings

The new **expected-round-score trump selector** won **30 of the 50** head-to-head comparisons, versus **19 wins for Ed Dakin's original suit heuristic**, with **one tie**. In the three-player competition, it won outright **29 matches**, versus **19** for the original picker; the RandomPlayer third competitor won outright once, and there was one shared-first tie.

| Quantity | New expected-score picker | Old heuristic |
| --- | ---: | ---: |
| Head-to-head wins | **30/50 (60%)** | **19/50 (38%)** |
| Head-to-head ties | 1 | 1 |
| Outright three-player wins | **29/50 (58%)** | **19/50 (38%)** |
| Mean accumulated final score | **95.94** | **92.68** |
| Mean score lead of new over old | **+3.26** | — |
| Median game margin, new minus old | **+8** | — |

The new selector has a favorable result in this *first* tournament, but **this is not yet strong statistical evidence it is generally better**. There are only **25 independent deal seeds**, each used for two correlated seat-swapped games. When comparing the two seat-swapped games as **one paired unit**, new had the higher two-game-average margin in **11 of 25 deal pairs**, old in **13**, and **1** paired unit was tied. The mean paired margin is still **+3.26**, but the paired-unit nonparametric bootstrap 95% percentile interval from 10,000 resamples is roughly **−1.76 to +8.80 points**. This interval includes zero. More deals and separately controlled policy RNG streams would be needed for a reliable conclusion.

## 1. Algorithm change

The repository's unused `alternateTrumpPicker(nplayers)` previously ran 50,000 older `mini_monte_sim()` bid-distribution simulations for each of four suits, then selected the suit whose **modal bid had the largest probability**. The active `pick_trumps()` method instead called the legacy `TrumpPicker.picktrump()` hand-statistics heuristic.

Now **both** `MonteCarloPlayer` and `SnakeMonteCarloPlayer` call their rewritten `alternateTrumpPicker()` **by default**. It delegates to `best_trump_by_expected_score()` in `alphawhist/core/WhistLib.py`. For each candidate trump suit `t` in hearts, diamonds, clubs, spades, the selector:

1. Samples **512 random feasible opponent deals/legal single-trick continuations** through `estimate_card_win_probabilities()`, producing each held card's estimated leading/following win probabilities under random opponents.
2. Computes an **exact probability mass function over the 0…N possible counts of won tricks** via `exact_bid_trick_pdf()` (the subset DP introduced earlier). The DP is exact under the approximate independent Bernoulli/random-card-order/leader-state model, **not** a solution to optimal Whist bidding.
3. Chooses the most probable prospective **legal** bid from that PMF. Only the already-known consecutive-zero prohibition can be enforced at this stage. **The dealer chooses trump before any bids and bids last**; the no-total-equals-number-of-tricks restriction cannot yet be determined.
4. Estimates the candidate suit's own round score as

   `E[tricks | t] + 10 * P(tricks = best_prospective_bid | t)`.

5. Selects the suit with the highest estimated score; exact ties break deterministically in suit order `h,d,c,s`.

The picker **does not** run 1,024 full decision rollouts for all four suits. The **1,024 default is unchanged for subsequent card decisions**. Both Monte Carlo agents keep their existing advanced bid selection once the actual trump and prior bids are known. They also retain different card-play objectives (own expected score versus score margin).

**Why expected score rather than hit-bid confidence?** Maximizing `P(T=b)` alone overlooks the expected number of tricks taken, each of which scores a point. The bonus of ten for an exactly matched bid and the baseline trick points are both relevant to the round's objective.

The original trump selector is preserved as `TrumpPicker.picktrump()` for `RandomPlayer` and for the `HeuristicTrumpMonteCarloPlayer` comparison subclass in the benchmark. The two tournament competitors differ **only** in their trump picker class methods; both bid and play cards with the same Monte Carlo implementation and 1,024 sampled continuations per nonforced decision.

## 2. Experimental design

- **50 complete 13-round games:** 55 tricks, 165 actual card plays, and 39 bids in each.
- **Three players per game:** new selector + old selector + `RandomPlayer`.
- **Two-game pairing:** each pair begins from the **same 13 predetermined initial deals**, with the new and old players swapping seats. Every three pairs rotate the seat combination `(0,1)`, `(0,2)`, `(1,2)`.
- **Seed:** base `20261031`; pair `j` uses `20261031 + j`; the dealer's random deal for a round is re-seeded to `pair_seed * 1009 + round_index`.
- **Both competing Monte Carlo agents:** 1,024 rollouts per card decision, one worker, same bidding code.
- **Win definitions:** head-to-head win is greater cumulative score than the **other** Monte Carlo agent (regardless of random's score); an outright three-player win requires sole highest final score.

Limitations: the dealt hands for a pair are common, but downstream stochastic bidding and play streams can diverge when trump decisions change. This is not a trajectory-identical or common-random-actions experiment. CPU wall time over all 50 games was about **254.2 seconds** on GitHub's shared CI runner, including the increased cost of evaluating four trump suits when the new agent is dealer. A different runner may have different timing.

## 3. Complete per-game results

The margin column is **new selector score minus old selector score**. “Winner” identifies the outright highest-scoring player; “Tie” means a shared first place.

| Game | Pair | Seed | New seat | Old seat | New score | Old score | Random score | New − old | Winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 20261031 | 0 | 1 | 106 | 98 | 91 | +8 | NEW |
| 2 | 1 | 20261031 | 1 | 0 | 103 | 71 | 91 | +32 | NEW |
| 3 | 2 | 20261032 | 0 | 2 | 81 | 98 | 46 | -17 | OLD |
| 4 | 2 | 20261032 | 2 | 0 | 107 | 79 | 39 | +28 | NEW |
| 5 | 3 | 20261033 | 1 | 2 | 107 | 90 | 78 | +17 | NEW |
| 6 | 3 | 20261033 | 2 | 1 | 77 | 100 | 48 | -23 | OLD |
| 7 | 4 | 20261034 | 0 | 1 | 121 | 83 | 91 | +38 | NEW |
| 8 | 4 | 20261034 | 1 | 0 | 119 | 72 | 84 | +47 | NEW |
| 9 | 5 | 20261035 | 0 | 2 | 87 | 110 | 38 | -23 | OLD |
| 10 | 5 | 20261035 | 2 | 0 | 118 | 85 | 82 | +33 | NEW |
| 11 | 6 | 20261036 | 1 | 2 | 93 | 109 | 33 | -16 | OLD |
| 12 | 6 | 20261036 | 2 | 1 | 103 | 102 | 60 | +1 | NEW |
| 13 | 7 | 20261037 | 0 | 1 | 82 | 105 | 18 | -23 | OLD |
| 14 | 7 | 20261037 | 1 | 0 | 97 | 75 | 33 | +22 | NEW |
| 15 | 8 | 20261038 | 0 | 2 | 84 | 91 | 80 | -7 | OLD |
| 16 | 8 | 20261038 | 2 | 0 | 89 | 69 | 67 | +20 | NEW |
| 17 | 9 | 20261039 | 1 | 2 | 98 | 127 | 20 | -29 | OLD |
| 18 | 9 | 20261039 | 2 | 1 | 127 | 118 | 20 | +9 | NEW |
| 19 | 10 | 20261040 | 0 | 1 | 137 | 101 | 47 | +36 | NEW |
| 20 | 10 | 20261040 | 1 | 0 | 95 | 101 | 59 | -6 | OLD |
| 21 | 11 | 20261041 | 0 | 2 | 103 | 91 | 71 | +12 | NEW |
| 22 | 11 | 20261041 | 2 | 0 | 52 | 97 | 86 | -45 | OLD |
| 23 | 12 | 20261042 | 1 | 2 | 109 | 118 | 58 | -9 | OLD |
| 24 | 12 | 20261042 | 2 | 1 | 106 | 101 | 58 | +5 | NEW |
| 25 | 13 | 20261043 | 0 | 1 | 63 | 137 | 65 | -74 | OLD |
| 26 | 13 | 20261043 | 1 | 0 | 118 | 73 | 64 | +45 | NEW |
| 27 | 14 | 20261044 | 0 | 2 | 124 | 91 | 50 | +33 | NEW |
| 28 | 14 | 20261044 | 2 | 0 | 66 | 101 | 58 | -35 | OLD |
| 29 | 15 | 20261045 | 1 | 2 | 93 | 85 | 57 | +8 | NEW |
| 30 | 15 | 20261045 | 2 | 1 | 121 | 85 | 59 | +36 | NEW |
| 31 | 16 | 20261046 | 0 | 1 | 89 | 97 | 49 | -8 | OLD |
| 32 | 16 | 20261046 | 1 | 0 | 88 | 88 | 69 | 0 | Tie |
| 33 | 17 | 20261047 | 0 | 2 | 82 | 111 | 62 | -29 | OLD |
| 34 | 17 | 20261047 | 2 | 0 | 99 | 85 | 71 | +14 | NEW |
| 35 | 18 | 20261048 | 1 | 2 | 77 | 70 | 78 | +7 | RANDOM |
| 36 | 18 | 20261048 | 2 | 1 | 63 | 93 | 89 | -30 | OLD |
| 37 | 19 | 20261049 | 0 | 1 | 106 | 82 | 77 | +24 | NEW |
| 38 | 19 | 20261049 | 1 | 0 | 111 | 96 | 78 | +15 | NEW |
| 39 | 20 | 20261050 | 0 | 2 | 108 | 96 | 31 | +12 | NEW |
| 40 | 20 | 20261050 | 2 | 0 | 62 | 87 | 46 | -25 | OLD |
| 41 | 21 | 20261051 | 1 | 2 | 69 | 88 | 68 | -19 | OLD |
| 42 | 21 | 20261051 | 2 | 1 | 97 | 78 | 90 | +19 | NEW |
| 43 | 22 | 20261052 | 0 | 1 | 71 | 114 | 50 | -43 | OLD |
| 44 | 22 | 20261052 | 1 | 0 | 99 | 62 | 34 | +37 | NEW |
| 45 | 23 | 20261053 | 0 | 2 | 96 | 103 | 66 | -7 | OLD |
| 46 | 23 | 20261053 | 2 | 0 | 106 | 70 | 69 | +36 | NEW |
| 47 | 24 | 20261054 | 1 | 2 | 102 | 94 | 89 | +8 | NEW |
| 48 | 24 | 20261054 | 2 | 1 | 103 | 92 | 70 | +11 | NEW |
| 49 | 25 | 20261055 | 0 | 1 | 84 | 67 | 54 | +17 | NEW |
| 50 | 25 | 20261055 | 1 | 0 | 99 | 98 | 68 | +1 | NEW |

## 4. Test and reproducing instructions

The new selector tests verify that:
- the choice is based on **expected score**, not merely the chance of hitting the most probable bid;
- two zero bids in succession remove the prospective bid-zero option;
- the **same new selector is called** by both Monte Carlo agent types;
- a comparison subclass still calls the **original heuristic**;
- full paired 13-round games complete, swap seats, obey rules, and emit a reproducible CSV.

CI run #41283551004 passed **430 tests**, including all older game, inference, rollout and scoring regressions.

Run locally:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m benchmarks.trump_tournament --games 50 --rollouts 1024 --seed 20261031 --csv tournament.csv
```

This is a **research baseline**, not a definitive trump-policy tournament. A natural follow-up is to test hundreds or thousands of *independent paired deals*, with explicit separate RNG streams for sampling/dealing/each opponent, and compare confidence intervals on scores, wins and match outcomes. Also benchmark the trump-decision computation itself: the exact DP is cheap, but estimating the win probabilities for **four suits** costs more than the legacy non-simulation heuristic.

## 5. Interpretation

This 50-game sample gives an initial **30–19 head-to-head lead** and **+3.26 average match points** for the new picker. However, the 25 deal-paired units favor the new picker only 11–13 with one tie, and the paired bootstrap interval contains zero. Thus the experiment **does not establish** a robust advantage, despite the favorable raw game-win count. This distinction matters when using paired-seed designs; treating all 50 games as independent binomial samples would exaggerate the evidence.

The default implementation was switched to the new expected-score picker as requested, with the original heuristic still available for comparison, and **future tournament results should determine whether the default remains preferable**.
