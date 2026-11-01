# Fast bid-aware policy for the last two tricks

**Experiment date:** November 1, 2026.  
**Implementation:** [`EndgamePolicy.py`](../alphawhist/core/EndgamePolicy.py) inside [`FastRollout.py`](../alphawhist/core/FastRollout.py).  
**Measured CI run:** [#41332606141](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41332606141).  
**Status:** Opt-in experiment. The production Monte Carlo policy still defaults to **`random`**.

## Motivation

The previous Monte Carlo evaluator uses uniform random continuation play for every agent after each proposed candidate card. The new `bid_aware_2` policy instead uses the acting player's *own two cards*, public trump, current pile, bid, and number of tricks won to estimate the value of playing either card first, then samples a choice from a softmax over the estimated future score. It does not access opponents' sampled hidden hands, and it never calls additional Monte Carlo rollouts.

Only when the acting player holds **exactly two cards and both are legally playable** does the heuristic run. It falls back to the standard uniform chooser for other hand sizes or forced moves. This is a policy for the final two tricks, not a new search algorithm.

The same algorithm supports **2–7 players**. The single-trick win probabilities are deliberately approximate and currently independent of opponents' hidden cards, play styles and past bidding likelihoods.

## Mathematical approximation

Let (b_i) be the player's bid, (w_i) tricks already won, (p_a) the modeled probability of winning the current trick by playing card (a), and (q_b^L) / (q_b^F) the modeled chance of winning the *forced* last trick with the remaining card (b), given that the player won/lost the current trick.

For each play order:

[
P(T=0)=(1-p_a)(1-q_b^F),quad
P(T=2)=p_aq_b^L,quad
P(T=1)=1-P(T=0)-P(T=2).
]

The choice utility is

[
Q(a)=P(T=1)+2P(T=2)+10P(T=b_i-w_i).
]

The acting player samples a legal action with a softmax temperature of **1.5 round-score points**. This is *not* a calibrated model of human or optimal player behavior.

### Fast implementation

- For `N=2..7`, precompute a tiny lookup table for estimated lead/follow trick-survival probability conditioned on card rank, trump status, opponents yet to play and whether opponents have one or two cards remaining.
- Only 52-bit card masks and public card IDs are used at runtime; the currently winning card in the public pile is checked explicitly.
- No `pow()`, combinatorial enumeration, simulated hidden-hand inspection or extra Monte Carlo calls occur in the hot two-card-choice calculation.
- Each sampled rollout runs to completion and uses exactly the same authoritative card rules and round scoring as the original random continuation.
- Both `MonteCarloPlayer` and `SnakeMonteCarloPlayer` can enable the experiment with `additionalParameters={"monteCarloPolicy": "bid_aware_2"}`; the default remains `"random"`.

## Tests

The CI suite passed **459 tests**, including new tests for:

- Correct probability-to-bid-bonus value conversion for needing zero, one, two, or an impossible number of tricks.
- Inability to beat an already-played unbeatable trump.
- Identical action values and identical stochastic selections when the opponents' hidden hands are swapped but the acting player's hand and public state remain identical.
- Mandatory follow-suit and forced-card behavior.
- Legal completion for all **2–7 player counts** across multiple seeded hands.
- **Identical seeded action-score vectors** with one versus two rollout workers and different batching sizes.
- Unchanged original random-policy evaluations when the opt-in option is omitted.
- Agent configuration acceptance and validation.

## Measured throughput and diagnostic quality

The following numbers come from [CI #41332606141](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41332606141) on GitHub-hosted Ubuntu. For each player count, the benchmark simulated **1,500 synthetic complete two-trick endgames** with identical dealt hands and random seeds across policies. Each run also evaluated **50 synthetic starting states** using 96 independently seeded random-policy continuations for each legal first card, as a *rough action-ranking reference*. The fixed hands are sampled from a standard deck; artificial bids and already-won trick counts are assigned for broad coverage, not to reproduce observed strategic opponent behavior.

| Players | Random two-trick runtime | Bid-aware runtime | Runtime factor | Best first-card agreement with random-continuation reference | Greedy score gain over random choice* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0081 s | 0.0119 s | **1.48×** | 46% | +0.084 |
| 3 | 0.0104 s | 0.0160 s | **1.53×** | 56% | +0.223 |
| 4 | 0.0129 s | 0.0208 s | **1.61×** | 54% | +0.170 |
| 5 | 0.0152 s | 0.0249 s | **1.64×** | 58% | +0.337 |
| 6 | 0.0177 s | 0.0289 s | **1.63×** | 60% | +0.475 |
| 7 | 0.0200 s | 0.0324 s | **1.63×** | 60% | +0.116 |

*The gain is from selecting the **greedy** highest-heuristic-value first card, evaluated against a **uniform random card** under sampled **random continuations** on the same complete hidden deal. The production policy uses a **softmax**, so the stochastic gains are smaller (roughly +0.02 to +0.23 modeled points across these small diagnostics). The 'oracle' is a noisy, complete-information *random-continuation* reference, not perfect play. These small positive diagnostic gains are not evidence of higher match win rate.*

A repeat pass on the same GitHub runner gave similar timings and exactly matching quality statistics. **Before precomputation**, the same two-trick simulator was approximately **2.7–2.9×** random runtime; the lookup-table optimization reduced this to approximately **1.5–1.6×**.

### Full seven-card evaluator

The [same CI run](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41332606141) also benchmarked the complete 256-rollout, three-player **seven-card early-round** action evaluator, including hidden-deal sampling, every candidate action, and all subsequent trick simulation. Median times over three runs:

| Continuation policy | Measured median time |
| --- | ---: |
| `random` | **0.05676 s** |
| `bid_aware_2` | **0.07084 s** |
| **Overhead** | **+24.8%** |

A second repeated run measured **0.05679 s** versus **0.07244 s** (**+27.6%**). The best action was not the same (`14s` for random vs `11c` for the bid-aware continuation). A different selected card is not sufficient to infer improvement.

**Performance assessment:** The policy is small and bounded in cost but **did not meet** our aspirational <10–20% full-rollout overhead target in this first version. This is acceptable as an **opt-in experimental baseline**; the global production default remains unchanged. More profiling and a proper paired tournament would be required before enabling by default.

## Reproduction

```bash
python -m pytest -q tests/test_endgame_policy.py tests/test_fast_rollout.py

python -m benchmarks.endgame_policy \
    --players 2 3 4 5 6 7 --states 1500 \
    --quality-states 50 --quality-rollouts 96 --seed 20261101 \
    --full-rollouts 256 --full-repeats 3
```

To enable the policy in a game:

```python
MonteCarloPlayer(
    0, additionalParameters={
        "monteCarloNumber": "1024",
        "monteCarloWorkers": "1",
        "monteCarloPolicy": "bid_aware_2",
    }
)
```

## Limitations and next experiments

The current lookup tables assume approximate independent rank/void probabilities, not true conditional probabilities from public card history. They do not use opponents' bids or previous revealed cards to estimate their hidden holdings, though the player's **own bid deficit** drives the utility.

The full rollout still models more than two cards with random choices. We have neither demonstrated calibration of the action probabilities nor improvement in full-match playing strength. The synthetic quality diagnostic is highly dependent on our specified random-continuation reference and a small sample of 50 states/player count.

Next useful steps: obtain actual endgame snapshots from tournaments; compare softmax choice probabilities against real actions of different opponent policies; improve bid-conditioned probability estimates; measure score and win rate against the untouched Monte Carlo baseline on a paired-deal multi-player tournament. Do not treat the first positive diagnostic gains as a statistically conclusive strength result.

## Follow-up: 6,000 legal game-endgame decision states

The small 50-state synthetic quality diagnostic above was substantially
less informative than a later
[6,000-state / 24.6-million-continuation benchmark](endgame-quality-6000-real-states-2026-11-01.md)
using **1,000 independently played rounds per player count**, 2–7 players,
with full public information-state observations and **2,048 sampled hidden
worlds per state**. On these states, the greedy two-card heuristic matched
the highest Monte Carlo-estimated card in **78.1–90.8%** of cases. Its
equal-weight mean expected-score regret was **0.099**, compared with
**0.308** for the existing softmax policy, **0.429** for a simple high/low
bid-aware rule and **0.576** for uniformly random choice. The evaluation
still assumes random future play and **does not prove stronger game play**;
the softmax temperature merits tuning. These follow-up results supersede the
earlier 46–60% synthetic decision-quality estimate for realistic endgame
states. The experiment code is preserved under
`benchmarks/endgame_quality.py`.
