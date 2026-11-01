# Monte Carlo performance optimization — 31 October 2026 (Pacific)

**Repository:** WhistyMcWhistface  
**Before:** `d3c588c4a24be3f9946e308706b6cbbec6cf4c5c` (full-game cProfile baseline).  
**Optimized change:** [PR #16](https://github.com/hugohadfield/WhistyMcWhistface/pull/16).  
**Environment:** GitHub-hosted Ubuntu Linux, CPython 3.11, single worker, 1,024 rollouts per non-forced card decision unless otherwise stated.  
**Evidence:** [same-run CPU/full-game comparison #41281916636](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41281916636) and [isolated reproducible evaluator comparison #41282042420](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41282042420).

## Summary

This pass targeted the hot spots identified in the earlier [full-game profile](full-game-profile-2026-10-31.md). It implements three optimizations:

1. The private simulation continuation path no longer checks a card's legality **twice**, while the public `SimState.play(card)` still validates card ownership and following suit.
2. Each sampled hidden world is converted from card strings to compact hand masks **once**, then independent candidate actions reuse cheap copies of the pre-encoded state. The batch uses reusable `random.Random` objects, reseeded to exactly the same deterministic seeds as before.
3. The advanced bidding step used by both Monte Carlo bot classes no longer runs **20,000 approximate Bernoulli trials per bid**. An exact subset dynamic program computes the trick-count probability distribution under the *same* random-card-order/leading-follows-winning model.

These changes were tested with exhaustive small-state bidding distributions and a seeded reference that reproduces the **old** rollout card scores exactly. In two same-run comparisons, full games took roughly **20–30% less wall time** for the ordinary Monte Carlo agent and **27–29% less** for Snake. Isolated identical-decision evaluators were about **1.36× faster (7 cards)** and **1.22× faster (3 cards)**.

The speedups are relative to the already-fast bitmask implementation—not the older pre-refactor object/deepcopy engine.

## 1. Detailed changes

### 1.1 Checked and unchecked simulated card transitions

Previously, `finish_round()` generated `legal_mask(hands[turn], pile)`, chose one of its set bits, and then called `SimState.play()`, which **again** validated ownership and following suit with an additional mask computation. Each simulated game included many thousands of such redundant checks.

Now:

- `SimState.play()` remains the **safe public entry point**; on an illegal card it raises without modifying the state.
- `SimState._play_legal()` performs the exact original card/trick state transition but assumes its input has already been validated.
- `finish_round()` constructs the legal mask, chooses a legal card, and calls `_play_legal()` directly.

For callable continuation policies, the evaluator still checks that a returned card is in the precomputed legal mask. Thus the optimization does **not** waive legality on user input or externally provided policy decisions.

### 1.2 Reuse per-world state and seeded RNGs

Previously each candidate card regenerated `SimState.from_snapshot(snapshot, sampled_world)`, repeatedly converting the same sampled card strings into bitsets. It also allocated a new `random.Random(seed)` for each candidate and another for each sampled world.

Now each world creates only one baseline `SimState`. Candidate actions receive independent, shallow copies of its **mutable** hands, pile and tricks-won lists; immutable bids and numeric metadata can be safely reused. Two persistent per-batch RNG instances are reseeded before world sampling and candidate policy continuation.

The seeds are unchanged (`_mix(seed, 2*index)` for each hidden deal and `_mix(seed, 2*index+1)` for candidate continuations). All candidates still receive the **same** hidden world and the same continuation RNG seed. Tests compare the optimized path against the original reference over multiple seeds and both built-in policies and assert **exact score-vector equality**.

Reusing seeded RNG objects eliminates allocations but **does not** remove the cost of reseeding; more aggressive RNG replacement would change the generated random streams and is left for a separate proposal.

### 1.3 Exact advanced bidding PMF

The previous active bidder used the following process: (i) estimate a lead/follow trick-win probability for every card using the existing 512-sample legal-play estimator, (ii) repeatedly take a uniformly random ordering of those cards, and (iii) draw Bernoulli wins for each card based on whether the bot will lead the trick. Winning means the bot leads next trick. It did this **20,000 times per round** to obtain a histogram of trick counts.

The new `exact_bid_trick_pdf(lead_probs, follow_probs, bid_position)` computes the exact marginal **for this approximating model** by dynamic programming over remaining cards `mask` and a Boolean lead state:

`F(mask, lead, k) = (1 / |mask|) * sum_{i in mask} [ (1 - p_i(lead)) * F(mask \ {i}, False, k) + p_i(lead) * F(mask \ {i}, True, k-1) ]`

with `F(0, lead, 0) = 1`. The cost is `O(2^n n^2)` for `n` cards and two lead states. Because this Whist schedule has **at most seven cards per hand**, this is cheap compared with the old 20,000-trial Python loop.

This removes Monte Carlo **variance in the bid-count distribution**, but it does **not** make the entire bidding model exact for real Whist. The initial per-card win probabilities are still approximate, future opponents are modeled using random legal play, and the estimator does not model all strategic dependencies. Crucially, the same bid-validity constraints (including the forbidden final sum and consecutive zero bids) are applied after obtaining the distribution.

The old `mini_monte_sim()` remains in the library for compatibility; both active Monte Carlo players now call `exact_bid_trick_pdf()`. The **1,024 default continues to apply only to card-decision Monte Carlo rollouts**. There is no longer a separate 20,000-simulation *distribution* budget in the active bidder.

## 2. Controlled, identical-decision evaluator comparison

To isolate card-simulator optimizations from changes in bids, hands and RNG consumption elsewhere, `benchmarks/measure_evaluator.py` evaluates an **identical snapshot** with a fixed seed, three repeated timings, and 4,096 rollouts per candidate set. It was run in both checkouts sequentially on the **same CI runner**. For each scenario, **all per-card expected-score vectors matched exactly**, including the selected best card.

| Position | Candidates | Old median | Optimized median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Seven cards, early round | 7 | **1.395317 s** | **1.029560 s** | **1.36×** |
| Three cards, mid round | 3 | **0.408548 s** | **0.333977 s** | **1.22×** |

Source: [CI run #41282042420](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41282042420). Times are median of **three**, which is useful for engineering triage but not a precision statistical estimate. `best_action()` was A♠ (`14s`) in the seven-card scenario and 2♠ (`2s`) in the three-card scenario, both before and after.

A key result here is **behavior preservation** for the card evaluator, independent of any full-match policy or random-deal changes. The bitmask simulation still uses the same checked `SimState.play()` when a real candidate action is submitted.

## 3. Full-game before/after comparisons

Both normal and Snake agents were tested on the same CI runner against two random players at the 1,024 rollout default. The tables below are **single game measurements**, not distributions over multiple seeds. Bidding changes the consumption of the global RNG and therefore subsequent deals, so the *resulting games and number of non-forced decisions can differ*. These are end-to-end performance comparisons of the workload, not bit-identical trajectory comparisons.

| CI run | Agent | Baseline full match | Optimized full match | Speedup |
| --- | --- | ---: | ---: | ---: |
| [#41281916636](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41281916636) | MonteCarloPlayer | 2.555 s | 2.065 s | **1.24×** |
| [#41281916636](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41281916636) | SnakeMonteCarloPlayer | 2.742 s | 1.996 s | **1.37×** |
| [#41282042420](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41282042420) | MonteCarloPlayer | 4.867 s | 3.797 s | **1.28×** |
| [#41282042420](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41282042420) | SnakeMonteCarloPlayer | 5.170 s | 3.657 s | **1.41×** |

Runner contention strongly affects absolute timings, which is why same-run measurements and the identical-snapshot comparison above matter.

### cProfile comparison

In run #41281916636, with a profiling agent in seat 0 and a fixed initial global seed:

- **Total instrumented wall time:** **9.452 s baseline → 7.067 s optimized**, around **1.34×**.
- **Advanced bid pipeline cumulative time:** **2.120 s → 0.381 s**, around **5.56×** faster. The remaining approximately 0.38 s largely belongs to the unchanged card-win probability estimation.
- **Card-rollout evaluator cumulative time:** **6.980 s baseline → 6.340 s optimized**, even though the optimized game had **37 non-forced rollout decisions** compared with **31 in the baseline game**. Do not treat this as a controlled per-decision benchmark; use Section 2 for that.

The profile also showed `mini_monte_sim()` consumed **1.753 s** in the old run, and the optimized active bidder no longer calls it. Both baseline and optimized used a single process worker for comparable CPU attribution.

The second run had higher hardware contention and yielded **16.304 s → 12.372 s** profiled wall time. Neither profiled wall time is representative of ordinary, noninstrumented game latency.

## 4. Correctness and test coverage

The optimized implementation passed **421 pytest cases** on the experiment branch.

New tests check:

- Every legal unchecked simulated card move gives **the identical state after every play** as the checked reference, for independent seeds and both random/highest continuation policies.
- For the same fixed seed and sampled worlds, all candidate score vectors from the optimized `_batch` are **exactly equal** to the original slow reference.
- The exact bid PMF agrees with an independently enumerated reference over **all card orderings and Bernoulli win/loss branches** in small hands, including both leading/following states.
- Deterministic probability edge cases (0/1), normalization, input validation, and card-order permutation invariance.
- The isolated evaluator timing benchmark returns reproducible card scores and supports early/mid fixed states.

Existing tests cover hidden-card privacy, follow-suit legality, all three player seats, complete 13-round games, partial tricks, serializing information states, score convergence, worker/batch reproducibility, and the 1,024-rollout default.

## 5. Reproducing and interpreting results

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Compare the same fixed state over independent timing repetitions:
python -m benchmarks.measure_evaluator --scenario early --rollouts 4096 --repeats 5
python -m benchmarks.measure_evaluator --scenario mid --rollouts 4096 --repeats 5

# Complete 13-round game, with and without the optional profiler:
python -m benchmarks.full_game --rollouts 1024 --bots MonteCarloPlayer SnakeMonteCarloPlayer
python -m benchmarks.profile_full_game --rollouts 1024 --limit 30
~~~

To reproduce the **baseline comparison** exactly, check out revision `d3c588c4a24be3f9946e308706b6cbbec6cf4c5c` separately and use the same seed and scenarios. The comparison CI action recorded the old/new side-by-side results but was removed from the permanent workflow after collection to avoid unnecessarily repeating it on every push.

### Further performance work

The exact bidding DP is now cheap. Remaining useful targets are `_choose_card()` and its random selection; repeated `random.Random.seed` calls still occur on every candidate even though objects are now reused. More aggressive optimization (for example a small custom deterministic PRNG, native loops, or vectorization) should be **separately tested** for statistical validity, parallel determinism, and actual playing-strength changes.

**Bottom line:** The optimizations preserve exact seeded card-evaluation behavior and substantially cut bidding overhead. The practical result is a **roughly 1.2–1.4× speedup** on representative whole-game CI runs, while isolated identical-state evaluations are **1.2–1.36× faster**. These numbers are measured and reproducible by the linked scripts; stronger claims about playing strength require independent multi-game evaluation.
