# Whist training-data collection: CPU profile (November 1, 2026)

## Scope and reproducibility

This is a **measured small-sample CPU profile**, not an extrapolation of
production-scale speed. We instrumented both pipelines:

- Full 13-round game capture (165 decisions at three players): one random-only
  game and one mixed random/MC-random/MC-bid-aware game, with **12 behavior
  rollouts** for the MC agents in the small benchmark.
- Neural teacher generation: **42 non-forced decision states**, stratified
  equally across 2–7 remaining cards, each with **256 Monte Carlo rollouts**,
  continuation policy `bid_aware_2`, compared with 1, 2, and 4 parallel
  state workers.

We used `cProfile` in one process for hot-function accounting, plus separate
**unprofiled** throughput runs and opt-in, constant-memory stage timers.
`cProfile` adds substantial overhead, so do not compare its absolute
runtime with normal production and do not add cumulative function times
that contain nested calls. GitHub-hosted runners have variable CPU
contention and core quotas.

The profiling workflow archives `.prof` files (read with `pstats` or
SnakeViz), top 35 functions sorted by cumulative and self time, per-stage
timing JSON files, a summary JSON, and exact worker-consistency checked
training shards. No GPU needed for this analysis.

Reproduce:

```bash
python -m benchmarks.profile_training_pipeline \
  --output-dir data/cpu-profile \
  --behavior-rollouts 12 --teacher-rollouts 256 \
  --states-per-size 7 --workers 1 2 4 --seed 20266117
```

## Finding 1: game observation reconstruction was unnecessarily expensive

Previously, `Game.observe(player_id)` reconstructed the complete public
event history for every nonzero player on every card decision. The old
`InformationState.apply_public_event` also created **two** fully validated
immutable snapshots per event. Every validation re-parsed every previous
public event (including earlier rounds).

Representative three-player **cProfile** numbers (same seed and scenario
on separate CI runs; figures are approximate, not a controlled speedup
study):

| Full-game metric | Before | After targeted fixes |
|---|---:|---:|
| All-random game total | 2.58 s | 1.04 s |
| All-random `Game.observe` cumulative | 2.05 s | 0.60 s |
| All-random JSON deserializations | ~590k | ~176k |
| All-random Python function calls | ~8.5m | ~2.5m |

The updated implementation caches each other player's observer-specific
state for the current round, applies only the public events added since its
last observation, validates cached history prefixes, and invalidates the
cache on a new deal or `resetGame`. Player policy state is always attached
fresh, not cached. `apply_public_event` now appends the validated event
as part of **one** dataclass replacement, instead of two.

A regression test compares the cached observation to an independently
implemented full replay **after every public event across three rounds**.
All game scores, legal-card masks, generated training traces, and teacher
labels remain subject to the usual CI regression suite.

Remaining overhead: `InformationState.__post_init__` still validates all
historical events for each immutable update, and public trace trimming and
serialization also traverse history. Further improvement would require
carefully preserving validation of externally supplied state; this is a
possible future optimization rather than justification for weakening
information-safety checks.

## Finding 2: teacher labeling is compute-bound, not I/O-bound

For 42 states at 256 rollouts per legal action, the unprofiled timed
single-worker breakdown on one CI runner was approximately:

| Stage | Accumulated wall time |
|---|---:|
| Monte Carlo evaluation | 1.065 s |
| Deserialize observer state | 0.006 s |
| Encode neural features | 0.0035 s |
| Validate/pack target counts | 0.0008 s |
| Compress and write shards | 0.0049 s |

The Monte Carlo phase represented about **98%** of the end-to-end time.
Compressing the existing 1,530-feature arrays was negligible at this
sample size, and rebuilding `InformationState` from a saved public
snapshot was not a major problem.

The **cProfile** hot call graph in the same label workflow was:

- `FastRollout.finish_round`: ~3.18 s cumulative of a ~4.13 s profiled run.
- `EndgamePolicy.pick_bid_aware_two_card`: ~1.59 s cumulative, called ~533k times.
- `FastRollout.SimState._play_legal`: ~0.89 s cumulative, called ~573k times.
- `InformationState.sample_hands`: ~0.32 s cumulative.
- RNG reseeding: around 50,600 `random.seed` calls; contributes a
  smaller but measurable fraction.

Thus the **next high-value optimizer is rollout continuation**—especially
the bid-aware two-card heuristic and per-card Python-level simulation,
not spending effort on NumPy feature encoding or file compression.
Sampling hidden hands was **not** the dominant bottleneck for this
particular three-player test.

No profiling result proves that an optimized rollout policy is better at
playing Whist. Any simulator optimization must preserve exact seeded
outcomes (or explicitly change and rebenchmark the policy).

## Finding 3: parallel CPU workers help, with diminishing returns

On one unprofiled CI run with the identical 42-state, 256-rollout dataset:

| Persistent labeling workers | Labeled states/sec |
|---|---:|
| 1 | 38.8 |
| 2 | 73.7 |
| 4 | 97.3 |

The same run verified **identical NumPy arrays** for all worker counts.
The throughput ratios are 1.90× at two workers and 2.51× at four workers,
relative to the one-worker run. This reflects fixed pool startup,
small work size and/or available runner resources; it does not justify
assuming linear throughput on a large server.

A subsequent CI runner showed different absolute speeds because of
machine contention, reinforcing the need for repeated measurements on
the intended training host. For a larger run, use 1, 2, 4, 8, 16
workers on the same machine, report CPU utilization, and choose the
best throughput.

## Finding 4: realistic gameplay can cost more than recorder overhead

On the mixed MC/random game, `WhistLib.estimate_card_win_probabilities`
accounted for about **3.0 seconds cumulative** in one ~4.4-second
profiled game, largely in trump selection and bidding. This is the
actual behavior-policy implementation, **not training trace I/O**.

Producing trajectories from stronger opponents will cost significantly
more than random-only trajectories, even if training labels are generated
offline. This suggests deliberately choosing the training mixture based on
state diversity and required playing strength rather than assuming all
games are equally cheap.

## Instrumentation APIs

The normal bot, generator and exporter have no timer allocations unless
the option is explicitly passed:

```bash
python -m benchmarks.generate_training_games \
  --games 20 --workers 4 --players 3 \
  --agents random,mc_random,mc_aware --rollouts 128 \
  --output-dir data/traces --timings-dir data/game-timings

python -m benchmarks.export_training_labels \
  --traces 'data/traces/*.jsonl' --output-dir data/labeled \
  --workers 8 --rollouts 1024 \
  --timings-json data/teacher-stage-timing.json
```

The time summaries contain counts, total, mean, max and approximate
power-of-two histogram p50/p95 bounds. Aggregated worker time is
**not the same quantity as total elapsed wall time**. The optional
timers introduce some overhead; profiling/benchmarking should be
turned off when generating production datasets.

## Priority order after this profiling exercise

1. **Done:** fix repeated whole-history replays for alternate observers,
   and redundant dataclass validation on public event append.
2. **Do next if performance becomes limiting:** optimize the
   `bid_aware_2` continuation and bitmask simulation hot path,
   preserving seeded outputs; benchmark on the real 1024-rollout teacher.
3. **Then:** profile `InformationState.__post_init__` history validation
   and the per-decision public trace serialization if game generation
   (rather than MC labeling) becomes the limiting phase.
4. **Do not prioritize yet:** feature-vector construction, compression,
   writing NumPy shards or GPU acceleration of this Python simulator.

We have **not** launched a production-scale dataset; this profile is
intended to identify priorities before committing that compute.
