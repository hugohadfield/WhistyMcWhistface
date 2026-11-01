# Do the trump pickers actually choose different suits?

**Measured:** November 1, 2026 UTC (October 31, Pacific), on the same 250-game trump-selector tournament originally reported in [the 250-game analysis](trump-choice-tournament-250-games-2026-11-01.md).  
**Runner:** GitHub Actions run [#41286613949](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41286613949).  
**Input:** Public `trump` events from all 250 complete, 13-round, three-player games (125 paired deal seeds; 1,024 Monte Carlo rollouts/card decision; four concurrent games).  
**Machine-readable data:** [CSV artifact (includes every trump choice and every match)](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41286613949/artifacts/13075718800), retained for 14 days.  
**Reproducible code:** [`benchmarks/trump_tournament.py`](../benchmarks/trump_tournament.py) and [`benchmarks/analyze_trump_choices.py`](../benchmarks/analyze_trump_choices.py).

## Main finding: they disagree on 38.56% of identical hands

The 250 games generated **3,250 actual trump declarations** (250 × 13 rounds). In each pair of games, the agents swap seats, while the exact same initial hands are dealt to each seat at each round. That provides a direct comparison when one of the *competing* Monte Carlo agents is the dealer:

- **1,084 matched dealer-hand/round cases** were eligible for the direct new-versus-old comparison.
- The algorithms picked **the same trump 666 times (61.44%)**.
- They chose **different trump suits 418 times (38.56%)**.
- In the remaining **541 matched dealer-hand/round cases**, the third `RandomPlayer` was dealer in both seat-swapped games; these were excluded from new-versus-old comparison. The original heuristic produced the same trump in both games on these 541 identical random-dealer hands, as expected.

**Conclusion:** The selectors are not merely different implementations of effectively the same trump rule. They make **substantially different actual decisions**, even though the tournament did not find a strong, stable difference in game-playing strength.

Importantly, these are **the actual declared trump suits in full games**, not hypothetical post-hoc calls with new Monte Carlo random seeds. The replay left all game decisions and random-state consumption unchanged except for reading the public event history when a game finished.

The replay's complete scoreboard **exactly matched the earlier 250-game run**: new picker 130 head-to-head wins, original heuristic 118, two ties, new minus old mean score −0.320, and mean final scores 93.336 versus 93.656. The replay reported 433 passing regression tests.

## Suit selection distributions on the 1,084 comparable hands

| Selected trump | New expected-score selector | Original Ed Dakin heuristic | New minus old |
| --- | ---: | ---: | ---: |
| Hearts (♥) | **290 (26.75%)** | **346 (31.92%)** | −56 |
| Clubs (♣) | **276 (25.46%)** | **243 (22.42%)** | +33 |
| Diamonds (♦) | **278 (25.65%)** | **303 (27.95%)** | −25 |
| Spades (♠) | **240 (22.14%)** | **192 (17.71%)** | +48 |
| **Total** | **1,084** | **1,084** | — |

The newer method selects suits fairly evenly, whereas the old heuristic exhibits a stronger preference for hearts and diamonds and selects spades least often. Suit-choice frequency alone **cannot establish disagreement**: the paired comparison above demonstrates that 418 actual *identical hands* elicited different suits.

## Exactly where they disagree (4×4 paired confusion matrix)

Rows show the **old heuristic**; columns show the **new expected-score picker** on the same dealer hand. Diagonal entries are agreements (sum = 666). Off-diagonal entries are disagreements (sum = 418).

| Original heuristic \ New selector | Hearts | Clubs | Diamonds | Spades | Row total |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Hearts** | **199** | 46 | 51 | 50 | 346 |
| **Clubs** | 33 | **155** | 24 | 31 | 243 |
| **Diamonds** | 29 | 55 | **186** | 33 | 303 |
| **Spades** | 29 | 20 | 17 | **126** | 192 |
| **Column total** | **290** | **276** | **278** | **240** | **1,084** |

For example, of the **346** cases where the heuristic chose hearts, the new selector also chose hearts **199 times** and chose a different suit **147 times**. The most common individual disagreement type was old diamonds → new clubs (**55 cases**), closely followed by old hearts → new diamonds (**51**) and old hearts → new spades (**50**).

## Disagreement by hand size

| Cards in hand | Comparable dealer hands | Different trumps | Disagreement rate |
| ---: | ---: | ---: | ---: |
| 1 | 84 | 34 | **40.48%** |
| 2 | 166 | 72 | **43.37%** |
| 3 | 166 | 70 | **42.17%** |
| 4 | 168 | 51 | **30.36%** |
| 5 | 166 | 58 | **34.94%** |
| 6 | 166 | 66 | **39.76%** |
| 7 | 168 | 67 | **39.88%** |
| **All** | **1,084** | **418** | **38.56%** |

The pickers disagree at every round length, including single-card hands. In this sample, disagreement is highest for two-card hands and lowest for four-card hands; this is descriptive and not proof of a systematic relationship.

## What this says about playing strength

The two selectors often choose different trumps, yet in the corresponding 250 full games:

| Tournament metric | New picker | Old picker |
| --- | ---: | ---: |
| Head-to-head wins | 130 | 118 |
| Final score mean | 93.336 | 93.656 |

The paired mean score margin was **−0.32 points per game**, with a paired-bootstrap 95% interval approximately **−2.46 to +1.81** (see [full tournament analysis](trump-choice-tournament-250-games-2026-11-01.md)). Across the combined 300-game sample (including the earlier 50), the mean advantage is approximately **+0.28** with an interval of roughly **−1.73 to +2.28**.

Therefore the weak playing-strength separation is **not because the two pickers usually select the same trump**. They disagree in nearly two of five matched cases. Under the other agents and assumptions of this tournament, the consequences of those differences tend to balance out.

### Caveats

- **Decision-state differences:** The same dealer receives the same cards across two paired matches, but their persistent consecutive-zero-bid eligibility may differ because their prior actions and outcomes can diverge. The new policy also includes stochastic 512-sample per-card probability estimation. The 38.56% rate measures **actual observed choice disagreement on matched hands**, not a controlled deterministic choice disagreement with all internal state and RNG held equal.
- **Seat swapping:** We exclude rounds where the third (random) player dealt, because neither competing picker made the trump decision in those two matches. These were 541 of 1,625 unique pair/round combinations.
- **Not causal:** A new-vs-old trump difference on a round does not by itself establish which pick was better. To evaluate *per-choice impact*, simulate matched downstream play for each candidate trump using a controlled, common-deal and common-policy-RNG experiment.
- **Original vs replay:** The original tournament CSV only logged final scores. This instrumented replay adds the 3,250 publicly declared suits while preserving the final outcome. The duplicate run took **724.252 seconds of tournament wall time** on a more heavily contended CI runner; unlike the original ~379-second 250-game run, this wall time should not be used to measure typical overhead from the logging code.

## Reproduction

```bash
python -m benchmarks.trump_tournament \
    --games 250 --rollouts 1024 --workers 4 \
    --seed 20261056 \
    --csv /tmp/replayed-250.csv \
    --trumps-csv /tmp/trumps-250.csv

python -m benchmarks.analyze_trump_choices --csv /tmp/trumps-250.csv
```

The full script prints total matched decisions, agreement/disagreement, chosen-suit marginal counts, the entire 4×4 cross-tab, seven hand-size strata and examples of differing selections. The one-time CI replay was removed from the normal push workflow after the analysis to avoid re-running hundreds of games on every commit.
