# Engineering change report — 31 October 2026 (Pacific time)

**Project:** WhistyMcWhistface, Python contract-Whist game and Monte Carlo agents  
**Repository:** https://github.com/hugohadfield/WhistyMcWhistface  
**Scope:** Work completed during the 31 October 2026 session (GitHub records the commits as 1 November 2026 in UTC).  
**Baseline:** the 2018-era implementation at `572e41ee1`.  
**Reviewed implementation:** `8028fe030586687ae37fc5ac4472619b41d2a19d` on `master`, before this report is added.  
**Verification:** [successful master CI run #41280274174](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41280274174): **373 tests passed** on Python 3.11.  
**Purpose:** Retain a technical record of the changes, supporting evidence, trade-offs, reproducibility instructions, and next steps.

## 1. Executive summary

Over this session, the repository evolved from a legacy Whist implementation with substantial rule/state bugs and expensive object-copy-based Monte Carlo play into a tested research platform with:

1. Corrected card, trick, bidding, hand, deal, trump-selection, and round-transition behavior, backed by a new pytest/GitHub Actions regression suite.
2. Immutable, versioned, JSON-serializable **observer information states**, recording only the observer's known cards and public events, with opponent card-possibility/void-suit constraints.
3. Independent counterfactual reconstructions and a constrained hidden-hand sampler, including fast uniform dealing for common unconstrained cases.
4. A separate, compact **history-free simulation engine** using integer card IDs and 52-bit hand masks instead of deep copies of the game and its event history for every rollout.
5. Paired Monte Carlo candidate comparisons (same sampled hidden world for each possible action), independently seeded RNG streams, optional process-level parallel batches, and integration with both the standard and Snake agents.
6. Complete 13-round integration tests, quantitative throughput/wall-time benchmarks, score-estimate convergence experiments, and a repeatable 10-match-versus-random benchmark.
7. A consistent **default of 1,024 rollouts per card decision** for both Monte Carlo bots and the standalone evaluator.

The latest CI verification has **373 passing tests**. In one GitHub Actions benchmark the new engine was approximately **58.1× faster** for a seven-card synthetic position (with 34.7× at three cards and 14.0× at one card). Another CI run measured approximately 66.9×/37.8×/13.5× for those positions. These are measured point estimates, not guaranteed speedups.

In the first 10 full games against two random opponents, **MonteCarloPlayer won outright 8/10**; **SnakeMonteCarloPlayer won 7/10**, tied first once and lost twice. These are promising smoke-test results, **not** enough games to establish their long-run playing strengths or statistically separate the algorithms.

## 2. Game rules, safety, and baseline regression work

The early session introduced `requirements-dev.txt`, `tests/test_whist_rules.py`, and a GitHub Actions test workflow. The following classes of pre-existing bugs were addressed:

- **Card comparisons and trick winning:** full multi-digit card ranks (10–14) are parsed correctly, and trump/lead-suit rules distinguish a winning trump from a non-trump. Card-beating helpers were repaired.
- **Legal play:** an actual move must be held in the player's **real** hand, not merely appear in its possibly-held candidate list. Following suit is enforced by legal-move generation.
- **Bids:** invalid bid types/ranges are rejected; the final bidder cannot make the sum of bids equal the cards available; the consecutive-zero-bid restriction is tracked and tested across rounds.
- **Round orchestration:** a game of AI-controlled players can deal automatically rather than entering the interactive/manual deal path. Dealer, leader, turn, and scoring transitions gained regression checks.
- **Trump selection:** empty suits no longer produce divide-by-zero in the selector; random players select trumps using the actual dealt hand.
- **Card state integrity:** mutable default arguments/hand aliases were removed, hands are copied at ownership boundaries, and real/possible-hand subset and uniqueness invariants are checked.
- **Determinization:** sampling of alternate hidden hands no longer assigns the same card to multiple opponent hands.
- **Bidding probability model:** `estimate_card_win_probabilities()` was replaced with legal-play-aware, non-overlapping random opponent deals that track the correct lead/follow positions and next leader. Its default is 512 samples; Monte Carlo agents' active advanced-bid calculation then uses a separate 20,000-simulation bid distribution estimator. **The 1,024 default discussed below applies to card decisions, not these bidding computations.**

These changes are represented by commits [`30b5ed645`](https://github.com/hugohadfield/WhistyMcWhistface/commit/30b5ed645), [`5c2a047a0`](https://github.com/hugohadfield/WhistyMcWhistface/commit/5c2a047a0), [`989ab7d4c`](https://github.com/hugohadfield/WhistyMcWhistface/commit/989ab7d4c), and [`e0d13af9b`](https://github.com/hugohadfield/WhistyMcWhistface/commit/e0d13af9b).

## 3. Observer information state and replay

**Main implementation:** `alphawhist/core/InformationState.py`, integrated in `alphawhist/core/WhistStructure.py` and `alphawhist/core/WhistPlayer.py`.  
**Milestone:** [PR #8](https://github.com/hugohadfield/WhistyMcWhistface/pull/8), commit [`6bf92a465`](https://github.com/hugohadfield/WhistyMcWhistface/commit/6bf92a465).

### What is stored

An `InformationState` is a frozen dataclass with schema version 1. It records the observer's seat and own remaining hand; game/round/phase; current dealer, leader, turn, trump and partial trick; public bids, tricks won, and scores; played cards; per-opponent remaining hand counts, candidate card sets, and known void suits; a strictly validated ordered public-event history; and a JSON-compatible policy-memory payload.

The public event types are `round_start`, `trump`, `bid`, `card`, `trick_end`, and `round_end`. Extra/private fields in an event are rejected. Inferring that a player is void in a suit uses their publicly visible failure to follow the led suit, **not** their actual secret hand. Played cards are removed from opponent candidate sets. Immutable snapshots can be encoded via `to_dict()`/`from_dict()`, `to_json()`/`from_json()`, or `save()`/`load()`.

The authoritative game maintains a player-0 information state incrementally. `Game.observe(player_id=0)` returns that view with current policy memory. Other seat IDs **are also supported now**: their belief state is reconstructed from their own current hand and the public round history, without accessing other players' real hands. This replay strategy is correct for the implemented observations but incurs overhead; it is not maintained incrementally for every seat.

### Independent counterfactual simulation

`Game.from_information_state(snapshot, seed=..., player_factory=...)` can build a separate game using sampled opponent hands and continue the current card-playing round, including a partly played trick. The original game object is not needed. Policy memory can be exported/restored with `export_policy_state()` and `load_policy_state()`; policy-specific extensions must implement their own hooks.

**Scope boundary:** the object-based reconstruction currently requires `phase == "playing"`. This is not yet a general purely functional `step(state, action)` engine or a complete arbitrary-phase game rewind.

### Hidden-hand sampler

`InformationState.sample_hands()` ensures hand sizes, candidate restrictions, known void suits, and card uniqueness:

- Identical opponent candidate pools: one shuffled allocation, **exactly uniform** over consistent allocations.
- Differing pools: up to 24 draws by independent uniform hand samples with collision rejection; accepted samples are uniform under those constraints.
- Highly constrained state: randomized bipartite matching ensures a feasible allocation where one exists, but **does not guarantee a uniform distribution**.

Neither consistency nor exact uniformity in some cases makes the belief model a calibrated strategic posterior: bids, opponent tendencies, and non-forced card preferences are not Bayesian-updated.

## 4. Fast Monte Carlo engine

**Main implementation:** `alphawhist/core/FastRollout.py`.  
**Milestone:** [PR #9](https://github.com/hugohadfield/WhistyMcWhistface/pull/9), commit [`9282e6590`](https://github.com/hugohadfield/WhistyMcWhistface/commit/9282e6590).

The old Monte Carlo card evaluator did approximately this **once per candidate move per trial**: deep-copy the entire `Game` and its players; resample opponents; replace simulation players; perform the remaining round through the normal, verbose game loop. The new public-event snapshots would have made that path even more expensive because authoritative state creation validates event histories.

The new engine deliberately separates those roles:

`Game / public history -> observer snapshot -> sampled world -> compact SimState -> simulated completion and round scores`

- **Cards:** integer IDs 0–51; each hand is a Python integer used as a 52-bit mask. Follow-suit checks and card removal are bit operations.
- **Simulation:** `SimState.play()` advances a legal card and awards a trick; `finish_round()` drives the rest of the round with a configured policy; `round_scores()` computes the existing bonus-based contract Whist scoring.
- **No event-log work in the hot path:** the fast simulator does not deep-copy the whole Game, serialize JSON or append public events for each simulated card.
- **Paired comparisons:** `evaluate_actions()` samples each feasible world once and tests **every legal candidate action** on that same world; the initial continuation-policy seed is shared across candidate actions.
- **Reproducibility:** per-rollout seeds are derived deterministically from a base seed and sample index, yielding the same numerical sums regardless of worker/batch count for the supported built-in policies.
- **Parallel execution:** batches can be sent through a `ProcessPoolExecutor` with `workers > 1`; the usual default is one worker. Small jobs can lose time to process startup and serialization.
- **Policies:** random legal future play is the default rollout policy. `highest` and custom callables are also supported (the latter must be picklable for process workers). This is an *evaluation model*, not an assumption that actual opponents will play randomly.

The old `Game.convertToDeterministicGame()`/object-based reconstruction path remains available for compatibility and comparison; the card-decision path in both AI bots uses the fast evaluator.

### Monte Carlo objective versus Snake objective

Given expected **round** scores `E[S_i | a]` for a candidate action `a`:

- **`MonteCarloPlayer`:** maximize `E[S_self | a]`.
- **`SnakeMonteCarloPlayer`:** maximize `E[S_self | a] - max_j E[S_j | a]` over opponents `j`.

The Snake calculation is **maximum of opponent expected scores**, not `E[max_j S_j]`. Both currently target the current round and do not explicitly optimize the probability of **winning the entire multi-round game** from accumulated scores. Both bots use the same advanced-bid logic.

## 5. Performance measurements

### Rollout throughput

Benchmark: `python -m benchmarks.rollouts --rollouts 80 --workers 1 --cards {7,3,1}`. The metric is completed remaining-round continuations per second, including setup, on a GitHub-hosted Ubuntu/Python 3.11 runner. The reference is the old object-oriented game path, not a separate native implementation.

| Cards remaining | Fast continuations/s | Old continuations/s | Speedup |
| --- | ---: | ---: | ---: |
| 7 | 13,350 | 230 | **58.1×** |
| 3 | 21,745 | 627 | **34.7×** |
| 1 | 19,508 | 1,393 | **14.0×** |

Source: [master run #41280274174](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41280274174). Another run, [#41280034897](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41280034897), measured about **66.9×, 37.8× and 13.5×**. Timings vary substantially with shared CI hardware, Python warmup, operating-system scheduling and the small workload; they are not guaranteed scaling factors.

### Full-match wall-clock time

A complete three-player match uses the round schedule `7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7`: **13 rounds, 55 tricks, 165 card plays, and 39 bids**. The benchmark includes dealing, bidding, card decisions, and scoring, with printing suppressed.

| AI | Rollouts per card decision | Full match: run #41280034897 | Full match: run #41280274174 |
| --- | ---: | ---: | ---: |
| Monte Carlo | 16 | 0.619 s | 1.060 s |
| Monte Carlo | 128 | 0.804 s | 1.461 s |
| Monte Carlo | **1,024** | **2.553 s** | **4.858 s** |
| Snake | 16 | 0.563 s | 1.032 s |
| Snake | 128 | 0.779 s | 1.435 s |
| Snake | **1,024** | **2.692 s** | **5.107 s** |

Each row is **one measured match on one CI invocation**, not a median over repeated independent games. In the separate 10-game-per-bot experiment below, total measured times were 27.55 s and 27.47 s respectively (about 2.75 s per game). Single-card latency has **not** been separately benchmarked; the full-game figures cannot be interpreted as a uniform per-action latency.

The faster rollout engine and its throughput are established within this workload. There is **no** demonstrated CPU-core scaling curve for multiple workers yet.

## 6. Score convergence and why the default is 1,024

**Artifacts:** `tests/test_rollout_convergence.py` and `benchmarks/score_convergence.py`.  
**Milestone:** [PR #11](https://github.com/hugohadfield/WhistyMcWhistface/pull/11), commit [`db53c9f7a`](https://github.com/hugohadfield/WhistyMcWhistface/commit/db53c9f7a).

The study evaluated each candidate card with nested, seeded rollout budgets of **16, 64, 256, 1,024 and 4,096**, comparing the estimated objective against an **8,192-rollout reference** using the same RNG prefix. The large reference is itself an estimate, **not exact ground truth**. Tests verify sample-prefix additivity, worker/batch reproducibility, lack of mutation, and convergence in aggregate against an analytically soluble one-card case. Individual Monte Carlo estimates need not converge monotonically.

### Three-card synthetic position

Cards and expected **round point** scores, including the 10-point contract bonus:

| Rollouts | 10♥ | 2♠ | A♠ | Selected card | Mean absolute error vs 8,192 reference |
| --- | ---: | ---: | ---: | --- | ---: |
| 16 | 4.562 | 3.500 | 6.500 | A♠ | 1.363 |
| 64 | 5.625 | 5.594 | 5.094 | 10♥ | 0.551 |
| 256 | 4.906 | 5.938 | 4.941 | 2♠ | 0.146 |
| **1,024** | **4.635** | **5.889** | **5.000** | **2♠** | **0.091** |
| 4,096 | 4.535 | 6.026 | 4.868 | 2♠ | 0.046 |
| 8,192 reference | 4.562 | 5.937 | 4.848 | 2♠ | 0.000 |

A choice based on 16 or 64 samples would have been different from the higher-budget reference, whereas all budgets from 256 upward chose 2♠ in this **one** position. The 7-card synthetic example likewise chose the reference-best A♠ at 1,024 and 4,096 samples, but at 64 samples selected 5♦ instead. The 7-card mean absolute objective error versus 8,192 was 0.117 at 256, **0.075 at 1,024**, and 0.045 at 4,096.

### Rationale for the 1,024 default

The default is now defined once as `DEFAULT_ROLLOUTS = 1024` in `FastRollout.py` and used by `evaluate_actions()`, `MonteCarloPlayer` and `SnakeMonteCarloPlayer`, including when their additional configuration is missing or empty. It is **not** a mathematically optimal sample size:

1. **Noise reduction:** very small samples clearly changed card scores and sometimes move selection. The increase from 256 to 1,024 cut the demonstrated 3-card MAE versus the finite reference from 0.146 to 0.091; in the 7-card case from 0.117 to 0.075.
2. **Decision robustness:** 1,024 agreed with the reference-best action in both examined synthetic positions. This is encouraging, **not** proof of a universal stabilization threshold; 256 also agreed in both positions.
3. **Acceptable runtime:** the new implementation plays an entire three-player 13-round match with one Monte Carlo agent and 1,024 rollouts per nontrivial decision in a few seconds on measured CI runs. Raising the count to 4,096 roughly quadruples rollout sampling work without an established corresponding improvement in win rate.
4. **Practical, explicit convention:** 1,024 is a convenient power-of-two experimental budget, not a game-theoretic requirement. Default worker count remains **1** and batch size **256**.

**Important distinction:** existing numbered JSON presets (e.g. `ThreePlayerMonte1000.json`) continue to use their explicitly configured counts. `whistmain.py` still starts three **random** players by default. Changing the Monte Carlo default did not change that entrypoint, legacy configurations, the 512-sample card-win estimator or the 20,000-trial advanced-bid estimator.

**Decision quality remains unproven:** we have *not* run enough independently seeded matches at 256, 1,024 and 4,096 to show that 1,024 is the best cost/strength trade-off, or computed standard errors/rank-confidence intervals across many real decision states. This is the next statistically meaningful optimization study.

## 7. End-to-end tests and 10-game performance against random opponents

**Artifacts:** `tests/test_full_game_ai.py`, `tests/test_monte_carlo_integration.py`, `tests/test_ten_games_vs_random.py`, `benchmarks/full_game.py`, and `benchmarks/ten_games_vs_random.py`.  
**Milestones:** [PR #10](https://github.com/hugohadfield/WhistyMcWhistface/pull/10), [PR #12](https://github.com/hugohadfield/WhistyMcWhistface/pull/12), [PR #13](https://github.com/hugohadfield/WhistyMcWhistface/pull/13).

The full-game tests cover each AI type in **all three seats** against random opponents. They check the correct schedule and full trick/card/bid totals; all hands emptied, scores and round transitions; per-round information-state consistency; and serialization after an entire match. The 55/165 totals were corrected after an initial test incorrectly expected 49/147. The test failure was an assertion mistake, not a broken played game. A separate experiment-runner smoke test checks outcome classification (sole winner versus shared first versus loss), seat rotation and seed assignment.

### One-time experiment: 10 games per AI, 1,024 rollouts

[Original CI log with all 20 per-game scores and summaries: run #41280034897](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41280034897).

Three seats; for game index `i = 0..9`, the AI occupied seat `i % 3` and the starting match seed was `20261031 + i`. Before each round, the deal generator was reseeded to give the same **starting hands** for corresponding games across the two AI versions. Random opponent moves can still diverge because the AI moves and game history diverge. All games were complete 13-round matches.

| Game | AI seat | Normal MC result (margin) | Snake result (margin) |
| --- | ---: | --- | --- |
| 1 | 0 | Loss (−21) | Loss (−17) |
| 2 | 1 | Win (+41) | Win (+35) |
| 3 | 2 | Loss (−12) | Loss (−20) |
| 4 | 0 | Win (+13) | Win (+12) |
| 5 | 1 | Win (+39) | Win (+57) |
| 6 | 2 | Win (+2) | **Tied first** (0) |
| 7 | 0 | Win (+4) | Win (+8) |
| 8 | 1 | Win (+43) | Win (+21) |
| 9 | 2 | Win (+17) | Win (+38) |
| 10 | 0 | Win (+7) | Win (+9) |

The margin is the AI's final cumulative score minus the **highest-scoring** random opponent.

| Outcome statistic | MonteCarloPlayer | SnakeMonteCarloPlayer |
| --- | ---: | ---: |
| Outright wins | **8/10 (80%)** | **7/10 (70%)** |
| Shared-first ties | 0 | 1 |
| Losses | 2 | 2 |
| Mean AI final score | 90.5 | 86.2 |
| Mean highest opponent score | 77.2 | 71.9 |
| Mean final-game margin | +13.3 | +14.3 |
| Total runtime for ten games | 27.55 s | 27.47 s |

With three equally strong symmetric players one would expect approximately one-third of outright wins if ties are rare, but **these are not statistically precise win-rate estimates**. For perspective, an ordinary Wilson 95% binomial interval for 8/10 is roughly **49–94%**, and for 7/10 roughly **40–89%**; these assume independent Bernoulli outcomes and do not account for the paired-deal dependence or possible seat effects. The observed one-game difference does **not** establish normal MC as stronger than Snake, and the larger mean Snake margin does not establish the reverse.

The 20-game trial was intentionally run once in the experiment PR. Its script remains in the repository, but the expensive experiment-only CI step was removed afterward so **every future push does not repeat all 20 matches**.

## 8. Tests, CI, and reproducibility

The project now uses `pytest` and an Ubuntu/Python 3.11 GitHub Actions workflow at `.github/workflows/whist-tests.yml`. The latest reviewed `master` run, [#41280274174](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41280274174), completed with **373 passing tests**. Coverage includes:

- Deck, suit, rank, trump, bidding, scoring, legal action and round-transition regressions.
- Card hand ownership, state consistency and feasible determinization.
- Immutable observer snapshots, strict public-event schemas, serializing/loading, mid-trick continuation, void-suit inference, no secret-opponent-hand leakage, and policy-memory round trips.
- Fast simulator rule equivalence, full-round scoring, legal following suit, parallel and batch determinism.
- End-to-end full games for both AI variants and all seats.
- Default 1,024-budget and explicit override behavior.
- Monte Carlo prefix consistency, exact analytically known expectations, score convergence and report/CSV output.
- Match-runner win/tie/loss classification.

**Run locally from the repository root:**

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Complete-match runtime at several card-decision budgets:
python -m benchmarks.full_game --rollouts 16 128 1024 --bots MonteCarloPlayer SnakeMonteCarloPlayer

# Remaining-round simulator throughput versus the legacy object engine:
python -m benchmarks.rollouts --rollouts 100 --workers 1 --cards 7

# Per-card expected-score convergence, including CSV export:
python -m benchmarks.score_convergence --scenario mid --rollouts 16 64 256 1024 4096 --reference 8192 --csv scores.csv

# Re-run the experimental win-rate comparison with fresh or matching seeds:
python -m benchmarks.ten_games_vs_random --games 10 --rollouts 1024 --seed 20261031
python -m benchmarks.ten_games_vs_random --games 100 --rollouts 1024 --seed 20261031
~~~

All CI timing measurements were taken on standard GitHub-hosted runners; **hardware-specific throughput and latency should be measured separately** on a target development machine. The win-rate script prints game-by-game scores, seating, seed, margin, outright win/tie/loss and aggregate totals.

## 9. Architectural limitations and risks

The implementation is significantly stronger but is not yet a calibrated optimal-play solver:

1. **Opponent model:** random legal play in rollouts, and the simpler advanced-bid simulation, may not predict strategic opponents. Estimating performance versus real opponents requires stronger opponent models.
2. **Uncertain hands:** forced card/void constraints are respected, but the matching fallback is not uniform and no posterior over bids/actions has been learned.
3. **Partial simulation API:** `Game.from_information_state()` can resume playing-phase states, not arbitrary bidding/dealing/scoring phases. The fast simulator only finishes the current round, not the rest of the 13-round game.
4. **Information boundary:** `InformationState` excludes other players' true hands, but a custom policy given an instantiated sampled-world `Game` or `SimState` can inspect the *sampled* private hands. Policies are not sandboxed behind a strict observation-only API.
5. **Historical validation overhead:** authoritative `InformationState` construction and event updates validate the stored event log, so ordinary full-game processing still has overhead; only the fast simulation inner loop is history-free.
6. **Game objective:** both card objectives estimate *current round* values, not probability of winning the match given accumulated scores. Snake computes max opponent expected score, not the expectation of the realized maximum.
7. **Statistical evidence:** the rollout-count convergence examples are synthetic and few; the 10-game win-rate study is small; CI timing comparisons are noisy; and direct head-to-head MC-versus-Snake tournaments have not been run.
8. **Reproducibility boundaries:** seeds control sampled worlds and the reported experiments; multiple process workers are tested for numerical equality. Full policy experiments still depend on Python's global RNG for some bidding and game decisions. In production, independent explicit RNG streams for each policy and each deal would be safer.
9. **Old compatibility code/documentation:** legacy `possible_hand` bookkeeping and object-based determinization still coexist with the newer observer states. Callers should use the new snapshot/evaluator APIs for new research.

## 10. Recommended next experiments

**Highest-value scientific experiment:** evaluate normal MC and Snake at budgets **128, 256, 512, 1,024, 2,048, 4,096** on hundreds or thousands of independently seeded, seat-balanced matches. Report win/tie/loss rates, score margins, uncertainty intervals, paired per-seed results, and compute per decision. This can identify where stronger card-value estimates stop improving actual playing strength.

**Next:** direct **MC-versus-Snake-versus-random** matches, and stronger bid-aware/opponent-aware future-play policies. Separately investigate optimizing **match-win probability** instead of current-round points.

**Engineering priorities:** (a) enforce observation-only policy interfaces, (b) improve constrained deal sampling and measure its distribution, (c) use explicit per-component RNGs, (d) profile authoritative event-history update overhead, (e) benchmark per-decision latency and multicore scaling on realistic mid-trick states, and (f) consider optimized/native rollout loops only if profiling justifies it.

## 11. Change landmarks and traceability

| Stage | Primary result | Reference |
| --- | --- | --- |
| Foundational rules and tests | Whist rules, hand integrity, Monte Carlo inference/bidding repairs | [`e0d13af9b`](https://github.com/hugohadfield/WhistyMcWhistface/commit/e0d13af9b41aab4113375fb899a48191305f9f69) |
| Observer snapshots | Privacy-preserving JSON states, event reducer, restoration | [PR #8](https://github.com/hugohadfield/WhistyMcWhistface/pull/8) |
| Fast simulation | Integer/bitmask engine, paired candidate trials, parallel reproducibility | [PR #9](https://github.com/hugohadfield/WhistyMcWhistface/pull/9) |
| Whole-game testing | All 13 rounds, all three AI seats, wall-clock timing | [PR #10](https://github.com/hugohadfield/WhistyMcWhistface/pull/10) |
| Score convergence | Rollout-count versus estimated card scores and chosen card | [PR #11](https://github.com/hugohadfield/WhistyMcWhistface/pull/11) |
| Default policy | Shared **1,024 card-decision simulations** default | [PR #12](https://github.com/hugohadfield/WhistyMcWhistface/pull/12) |
| Playing-strength smoke test | 10 complete matches against random opponents per bot | [PR #13](https://github.com/hugohadfield/WhistyMcWhistface/pull/13) |

**Bottom line:** The repository is now suitable for reproducible small-to-medium Monte Carlo algorithm experiments and has comprehensive regression coverage for the currently supported game modes. The **1,024** default is a **pragmatic, evidence-informed baseline**, not a proven optimal policy budget. Fast simulation, unbiased handling of many hidden-hand cases, and favorable results versus random agents are real achievements; reliable *comparative playing-strength* conclusions require larger and more rigorous experiments.
