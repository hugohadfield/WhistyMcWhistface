# Whist neural training data: real traces and Monte Carlo labels

The infrastructure separates three different sources of information:

1. **Real play:** authoritative game outcomes and public decisions.
2. **Observer features:** exactly what an individual player could know at
   its decision point, via `Game.observe(player_id)`. No opponents' real hands.
3. **Counterfactual teacher:** sampled *possible* hidden worlds and the
   resulting distribution of **additional tricks won** under a specified
   continuation policy. This is *not* the original real deal.

The supervised target for the existing `DistributionalWhistMLP` is
`P(additional_tricks=k | observer_information, candidate_card)`, for
`k=0..7`. The eight-bin histograms reconstruct the original MC expected
score via `previous_tricks + E[additional_tricks] + 10 *
P(previous_tricks + additional_tricks == bid)`.

## Example pipeline

First collect **genuine full-game decisions** from a mixture of opponents
with independent per-game and per-round seeds:

```bash
python -m benchmarks.generate_training_games \
  --games 20 --players 3 --workers 4 --seed 2026411 \
  --agents random,mc_random,mc_aware --rollouts 128 \
  --output-dir data/traces-v1
```

This writes `game-0000000.jsonl`, etc. Each file contains a game header,
13 public round starts, 165 real decision rows (three players × 55 card
plays), completed round scores and a final game summary. All three players
contribute supervised examples, including forced moves (which the labeling
step skips by default).

For each decision, retain an observer-specific `InformationState`, legal
cards, selected real action, behavior/continuation policy information,
game and deal IDs, actual final tricks/score, and an exact-bid indicator.

Then generate **counterfactual MC labels** offline:

```bash
python -m benchmarks.export_training_labels \
  --traces 'data/traces-v1/*.jsonl' \
  --output-dir data/labeled-v1 \
  --rollouts 1024 --policy bid_aware_2 --workers 4 \
  --state-batch-size 4 --prefetch-batches 2 \
  --shard-size 256 --seed 2026441
```

For a cheap end-to-end pilot, add `--max-states 100 --rollouts 64`.
This only limits the labelling workload; it does not change the saved game
traces.

The training tensor output consists of compressed NumPy `.npz` shards
and a JSON manifest. Each shard contains:

| Tensor | Type / shape | Meaning |
|---|---|---|
| `features` | float32 [N,1530] | Information-safe neural input |
| `legal_mask` | bool [N,52] | Only legal cards are scored |
| `trick_counts` | uint32 [N,52,8] | MC count for each card and remaining trick outcome |
| `n_rollouts` | int32 [N] | Label confidence/budget |
| `remaining_tricks`, `tricks_won`, `bids` | int32 [N] | Exact score reconstruction |
| `chosen_card_id` | int32 [N] | Observed real action, not the teacher argmax |
| `actual_remaining_tricks`, `actual_round_score` | int32 [N] | Outcome for real chosen action only |
| `game_id`, `round`, `decision_in_round`, `player`, `teacher_seed` | array | Reproducible provenance |

Counts for illegal cards are **zero**, and every legal card histogram
sums to `n_rollouts`. Divide by `n_rollouts` to get the soft-label
probabilities used by `distributional_loss()` in
`alphawhist.learning.model`. All other states and action types are masked.

## Split integrity

Train, validation, and test splits use a stable hash of a `deal_group`
identifier, **not individual decisions**. All decisions in a complete
game fall in one split. Two seat-swapped replays of the same dealt
head-to-head match also share the *same group*, preventing leakage
between those paired trajectories. Approximate split fractions are
80%, 10%, 10%; small experiments can have empty validation/test splits.

The output directory must be empty to avoid accidentally mixing datasets.
Every completed `.npz` shard is written via a temporary file followed by
an atomic rename; the manifest is written only after all labeling completes.
Already-written shards survive many kinds of interruption, but there is no
automatic resumption of a partially generated dataset. To continue a
dataset, run a new output directory and merge only after validating
metadata and split-group integrity.

## Head-to-head tournament tracing

To record observations from strong-bot competitions instead of separate
training games, enable the optional tournament flag:

```bash
python -m benchmarks.endgame_policy_tournament \
  --games 20 --rollouts 1024 --workers 4 --seed 2026412 \
  --csv tournament.csv --training-dir data/tournament-traces
```

Each game process writes a different file, so workers do not contend
over one JSONL stream. Rejected/unsupported and forced decisions remain
auditable in the raw trace, but the default label exporter skips forced
decisions because there is no meaningful card choice.

## Optional, privileged replay archive

**Do not train on hidden opponents' actual hands.** The trace recorder
does not include them. If debugging exact dealt-game reconstruction is
required, enable a *separate* explicit privileged archive:

```bash
python -m benchmarks.generate_training_games \
  --games 2 --players 3 --agents random --rollouts 8 \
  --output-dir data/public --private-dir data/privileged
```

The privileged archive contains all true initial hands; limit access to
it and never pass it to the exporter/encoder. The public trace contains
only a SHA-256 fingerprint of the deal and the observer state.

## Performance and correctness

- Regular games incur no extra `observe()` calls or recording when no
  recorder is attached. MC score-only runs still use the existing
  score-only evaluation path; extra histograms are opt-in.
- Labeling consumes extra Monte Carlo for **every non-forced card
  decision**, potentially far more compute than generating game traces.
  Use cheap collection runs and progressively larger teacher budgets.
- The label exporter now uses **one persistent process pool for the entire
  dataset**, parallelizing across independently seeded decision states.
  `--workers N` controls total CPU processes; **each process evaluates its
  own states with `evaluate_actions(workers=1)`**, with no nested pools.
  `--state-batch-size 4` amortizes interprocess communication and
  `--prefetch-batches 2` bounds queued work. At most
  `workers * prefetch_batches * state_batch_size` decision records are
  submitted, regardless of dataset size. Tune the state batch size for
  your machine; smaller batches help load-balance costly early rounds.
- Results are emitted in deterministic input order, so the same seeds,
  teacher budget, and trace inputs produce identical training arrays and
  shard boundaries for one worker or many. Monte Carlo per-action
  **common random numbers are unchanged**, and training/validation/test
  split assignment remains by whole deal group.
- The exporter prints `states_per_second` and elapsed wall time to make
  real throughput benchmarking straightforward. CI does not require a
  speedup on GitHub-hosted runners, which can have limited CPU cores.
  For a 16/32-core CPU, start with `--workers 12` or `--workers 28`
  and check actual throughput before allocating every core.
- Training labels are sampled under a particular future-play policy and
  are not ground-truth perfect-play values.
- Initial deals are reset per round to remove coupling between card
  draws and the number of random values used by earlier policy decisions.
- Tests assert that histogram counts reconstruct the unmodified MC score
  estimates, that worker/batch configurations produce identical seeded
  results, and that private opponent hands cannot leak into the neural
  observation.

## Recommended initial sequence

Collect a pilot of several dozen mixed-policy matches and export a few
thousand non-forced decisions using 64–256 teacher rollouts. Train the
316k-parameter model and validate input/target behavior. Then scale
game collection towards 2,000 independent games and label a carefully
selected subset at 1,024+ rollouts before investing in much larger runs.

All datasets should carry code commit, feature/trace schema version,
teacher continuation policy, sampling seed, rollout count and split seed.
The trained model requires PyTorch; the trace and teacher generators do not.
