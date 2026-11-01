# Full-match cProfile: 1,024-rollout Monte Carlo (31 October 2026, Pacific)

**Game:** 13 rounds, 55 tricks, 165 real card plays; one `MonteCarloPlayer` in seat 0 versus two `RandomPlayer` opponents.  
**Configuration:** `--rollouts 1024 --seed 20261031 --seat 0`; Python 3.11, one GitHub-hosted Ubuntu runner, one process worker.  
**Profiling tool:** Python standard-library `cProfile` / `pstats`; gameplay printing redirected to `os.devnull`.  
**Raw evidence:** [Successful experiment CI run #41281313795](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41281313795); [download the binary cProfile artifact](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41281313795/artifacts/13073427054) (GitHub retained artifact for 14 days).  
**Script:** `benchmarks/profile_full_game.py`.  
**Correctness:** Same game completed with cumulative scores `(97, 83, 85)`; 379 pytest cases passed in the profiling PR.

## Executive findings

The instrumented match took **12.457 seconds wall clock** and reported **12.423 seconds of summed cProfile self time** over 38,230,671 function calls. `cProfile` instrumentation adds substantial overhead: in earlier **unprofiled** CI measurements of the same match configuration, the match took approximately **2.55–4.86 seconds**, depending on the runner invocation. **12.46 seconds is not an estimate of ordinary play latency.**

| Workload | Inclusive CPU seconds reported | Fraction of profiled total | Notes |
| --- | ---: | ---: | --- |
| `evaluate_actions`: non-forced Monte Carlo card decisions | **9.272** | **74.6%** | 31 calls; includes sampling, all candidate continuations and scoring |
| `getPlayerBids`: bidding | **2.811** | **22.6%** | 13 calls, includes the Monte Carlo agent's advanced bid estimator |
| Other game activity | ~0.34 | ~2.8% | Deal, scoring, ordinary random plays, public event handling, and other overhead |

**Do not sum the individual hotspot cumulative times in the following sections.** Cumulative timings are *inclusive* of called functions, and the public event/replay costs appear under gameplay and bidding too.

There were **55 real Monte Carlo agent card plays** (one per trick), of which **31 had multiple legal candidates** and invoked the evaluator. The other **24 moves were forced** and therefore did no rollout search. The evaluator sampled **31 × 1,024 = 31,744 hidden deals** and evaluated **106,496 candidate continuations** in total (an average of 3.35 actions per searched decision). These continuations produced **1,212,416** calls to the `SimState.play` card transition method and **1,105,920** calls to the simulated follow-up card chooser.

## Hotspots in the simulation backend

These measurements are from `cProfile` on the same 12.423-second instrumented run.

| Function | Calls | Cumulative time | Self time | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `FastRollout.evaluate_actions` | 31 | **9.272 s** | 0.001 s | All non-forced card evaluations |
| `FastRollout._batch` | 124 | 9.270 s | 0.318 s | Hidden deals plus all candidate branches |
| `FastRollout.finish_round` | 106,496 | **6.029 s** | 0.853 s | Simulation loop and repeated move processing |
| `FastRollout.SimState.play` | 1,212,416 | **2.729 s** | **1.875 s** | Validation, bitmask mutation and trick completion |
| `FastRollout._choose_card` | 1,105,920 | **2.241 s** | 0.769 s | Uniform random legal future action |
| `InformationState.sample_hands` | 31,744 | **0.943 s** | 0.144 s | Observer-consistent hidden-card sampling |
| `FastRollout.SimState.from_snapshot` | 106,496 | **0.774 s** | 0.276 s | Creates a fresh independent branch per action |
| `FastRollout.legal_mask` | 2,318,367 | 0.382 s | 0.382 s | Follow-suit legality bit operations |
| `FastRollout.trick_winner` | 437,248 | 0.345 s | 0.345 s | Winner calculation at trick completion |

Python's `random.randrange` appears **2,212,950** times with **2.407 s cumulative time** (overlapping move choice and bidding), and `random._randbelow_with_getrandbits` appears **2,581,141** times with **1.629 s cumulative**. Also notable:

- **138,240 `random.Random` constructor calls**, accounting for ~0.875 s cumulative in its constructor and 0.821 s in Python's `seed` wrapper (overlapping). This happens because the implementation constructs a fresh continuation-policy RNG per candidate branch and a hidden-deal RNG per sampled world. Seed generation alone has measurable cost.
- **2,318,367 legal-mask calls** and **1.21 million transition calls**: Python per-card function call and validity-check overhead is visible even with bitmask hands.
- **106,496 snapshot-to-simulator reconstructions**: every candidate action gets a fresh `SimState`; this is deliberate for independence, but its allocations and card conversions cost time.

The profiler confirms the bulk of the compute is actual Monte Carlo work, not the old deep-copy/JSON replay behavior.

## Bidding and authoritative information state

| Function | Calls | Cumulative time | Self time | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `WhistStructure.getPlayerBids` | 13 | **2.811 s** | ~0 s | All bids across 13 rounds |
| `MonteCarloPlayer.makeAdvancedBid` | 13 | **2.762 s** | ~0 s | Normal AI bidding (same method as Snake) |
| `WhistLib.mini_monte_sim` | 13 | **2.291 s** | **0.924 s** | The active 20,000-draw advanced-bidding estimator |
| `WhistLib.estimate_card_win_probabilities` | 13 | **0.470 s** | 0.063 s | Legal-play-based card win probability estimator |
| `WhistStructure._record_public_event` | 285 | 0.375 s | ~0 s | State reducer and public event updates |
| `InformationState.apply_public_event` | 285 | 0.375 s | 0.003 s | Immutable public event transition |
| `InformationState.__post_init__` | 615 | **0.383 s** | 0.049 s | Validation of immutable information state |
| `Game.observe` | 32 | 0.021 s | ~0 s | Snapshot queries in this seat-0 match |

The **bidding calculation accounts for roughly one quarter** of profiled match time even though it is invoked only once per round by the AI. This becomes relatively more important if future optimizations reduce card-rollout cost.

The **public history validation cost (~0.375 s inclusive in the event transition chain)** is now a secondary bottleneck, not the first place to invest. The fast inner simulator does not record one public event per simulated move.

## Recommended optimization order

The following are opportunities suggested by the profile, **not changes performed by this experiment**. Each should be benchmarked before and after, with card legality and seeded score equivalence tests.

1. **Fast-simulation transition loop:** Make a specialized checked/unchecked transition split. Validate candidate actions at entry to a branch, then avoid repeating expensive defensive validation and Python method dispatch on every internally generated legal card. Keep the safe public API. The ~1.875 s in `SimState.play` self time makes this a clear priority, but exact speedup is unknown.
2. **Reduce RNG initialization:** Avoid reconstructing and reseeding a `random.Random` object for every candidate continuation where equivalence permits a cheaper indexed deterministic stream. Preserve identical hidden worlds between candidate actions and reproducibility across workers. Because RNG affects tie-breaking and trajectories, test statistical equivalence rather than assuming bit-identical results after changing streams.
3. **Optimize random legal-card selection:** The roughly 1.1 million `_choose_card` calls, plus Python `randrange`/`_randbelow`, justify prototyping faster uniform selection from small bitsets, perhaps by caching legal actions or improving batched sampling. Correct distribution matters more than making moves deterministic.
4. **Profile and simplify bidding:** `mini_monte_sim` is ~2.29 s cumulative and is called once per round. Explore exact/DP bid-probability calculations and/or an adaptive sample budget. The 20,000 bid draws are **separate from the 1,024 card-rollout default**.
5. **Reduce repeated branch setup:** Cache immutable encoded card/constraint fields per observer snapshot; copy compact hand-mask vectors rather than repeatedly converting card strings for every candidate. Current branch reconstruction accounts for ~0.774 s inclusive.
6. **Public-state validation:** Validate *new* events incrementally while maintaining strict schema and invariants; avoid re-validating old JSON events on every authoritative transition. This is only ~3% of measured profiled time here, so lower priority.
7. **Parallel scaling and latency:** Measure true wall clock without `cProfile` at a range of `workers` and position complexities. Standard `cProfile` of a single parent process does **not** attribute CPU time spent inside child workers; this specific run deliberately used **one worker**.

## Reproducing the profile

~~~bash
# From repository root, one complete game, baseline agent:
python -m benchmarks.profile_full_game --bot MonteCarloPlayer \
    --rollouts 1024 --seed 20261031 --seat 0 --limit 35 \
    --output profile-full-game.prof

# Compare the Snake scoring objective:
python -m benchmarks.profile_full_game --bot SnakeMonteCarloPlayer \
    --rollouts 1024 --seed 20261031 --seat 0 --limit 35 \
    --output profile-snake.prof

# View the saved binary profile locally using only Python:
python -m pstats profile-full-game.prof
# In the pstats prompt:
#   sort cumulative
#   stats 30
#   sort time
#   stats 30

# Uninstrumented wall time (not comparable to profiler-inflated execution):
python -m benchmarks.full_game --rollouts 1024 --seat 0 \
    --bots MonteCarloPlayer
~~~

The profiler's `--output` writes a reusable binary cProfile record; the CI run uploaded a copy as a temporary artifact. The script is checked by `tests/test_full_game_profiler.py`, including a full-game smoke test for each AI class at a low rollout budget.

### Scope and interpretation

This is **one seeded, full, three-player game**, not a statistical distribution of latency across seats/deals/machines. The workload mixes early- and late-round decision sizes, and results are dominated by the AI's actual number of legal candidate cards and the remaining trick count. The percentages and absolute times can differ for other hands, seeds, and especially multiworker runs. Use ordinary unprofiled benchmarks for user-facing latency, cProfile self/cumulative timings for prioritizing improvements, and independent game-quality experiments to ensure optimization does not harm decisions.

**Bottom line:** Simulation is the main bottleneck (~75% of profiled compute), **bidding second** (~23%), and observer/public-state validation currently only a few percent. The immediate code hot spots are the **simulated card transition**, **uniform random continuation move selection**, and **per-branch random-state initialization**. This supports optimizing the fast simulator's inner loop next rather than rewriting the public-state layer first.
