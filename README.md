
# WhistyMcWhistface - A contract whist platform for AI research

WhistyMcWhistface is an implementation of the rules of contract whist along with useful functionality for writing bots to play the game.

Two Monte Carlo agents are provided: MonteCarloPlayer maximizes expected own-round points, while SnakeMonteCarloPlayer maximizes its expected lead over the strongest expected opponent. Both support a default of 1,024 sampled rollouts per card decision.

Some limited GUI functionality is provided via the easygui library

TrumpPicker was written by Ed Dakin

# Getting started

The simple entrypoint is whistmain.py, which currently loads three random players. Configure MonteCarloPlayer or SnakeMonteCarloPlayer to run AI-vs-random games.



## Engineering change report — 31 October 2026

For an end-to-end account of the rules and state refactor, fast simulation,
full-game tests, measured performance, rollout convergence, the rationale
for the **1,024 rollout default**, and AI-vs-random results, read the
[31 October 2026 engineering report](docs/engineering-report-2026-10-31.md).

For a measured **full-game CPU profile** (simulator, bidding, RNG, public
information updates), see the
[full-game profiling results](docs/full-game-profile-2026-10-31.md).
Reproduce with `python -m benchmarks.profile_full_game --rollouts 1024`.

**Follow-up optimization:** Card-rollout loops now avoid redundant checks,
reuse pre-encoded candidate states and seeded RNGs, while bidding uses an
exact subset dynamic program instead of 20,000 sampled bid outcomes.
The [optimization results and before/after measurements](docs/rollout-and-bidding-optimization-2026-10-31.md)
show approximately 1.2–1.4× faster whole games and **identical seeded
card-evaluation scores**. You can run the fixed-position benchmark with
`python -m benchmarks.measure_evaluator --scenario early --rollouts 4096`.


## Trump selection: expected-score picker (November 2026)

Both Monte Carlo bots now actively choose trumps by evaluating all four suits
with 512 samples of card win probabilities and the exact trick-count subset DP,
then maximizing modeled expected round score (expected tricks plus the
10-point bonus probability for the best provisional bid). The dealer picks
trumps before bids have been placed, so the final-bid no-exact-sum
restriction cannot yet be predicted by this model.

The original `TrumpPicker.picktrump()` heuristic remains available, and the
[50-game paired-seat tournament report](docs/trump-choice-tournament-2026-10-31.md)
compares the two using identical 1,024-rollout bidding/card-play policies.
The new version won **30** head-to-head games versus **19** for the heuristic,
with **one tie**, but the 25 paired deal-level results are inconclusive.
Reproduce with
`python -m benchmarks.trump_tournament --games 50 --rollouts 1024`.

To run independent games concurrently (separate CPU processes), use e.g.
`python -m benchmarks.trump_tournament --games 500 --rollouts 1024 --workers 4 --csv tournament.csv`.
Each bot still uses one CPU worker for its per-decision Monte Carlo rollouts;
`--workers` controls how many **whole games** run concurrently. Results and
CSV rows remain ordered by game, with reproducible scores for fixed seeds.
In the summary, `wall_seconds` is the actual end-to-end tournament time;
`total_seconds` adds up match times and is **not** elapsed wall time with
parallel execution.

## Serializable information state and rollout experiments

The authoritative hard-constraint beliefs of **player 0** are tracked
incrementally and independently of other players' actual hands; observations
for other seats are reconstructed from their own cards and public history.
Observations are immutable, versioned, JSON-serializable and updated using
public events. Each snapshot includes the observer's remaining hand,
per-opponent candidate cards
and remaining counts, known void suits, current bids/scores/trick/turn, full
public event history, and a JSON-compatible policy-memory payload.

```python
from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistStructure import Game

# During an in-progress game, after dealing (including midway through a trick):
snapshot = game.observe(player_id=0)
snapshot.save("decision_42.json")

# Another process/machine can discard the original Game entirely.
restored = InformationState.load("decision_42.json")
for seed in range(1000):
    rollout = Game.from_information_state(restored, seed=seed)
    action = rollout.players[rollout.turn].getRealPossibleMoves(rollout.pile)[0]
    rollout.playPartialTrick(action)
    rollout.playPartialRound()
    print(rollout.scores)
```

**Security/inference boundary:** Neither `InformationState` nor its JSON
contains the true hands of opponents. A rollout samples a fresh, consistent
assignment given the snapshot's hard constraints. Undealt cards need not
be assigned to any player. Every rollout is independent of the original game
object; a different `player_factory(player_id, sampled_hand)` can supply
alternative algorithms. Each worker should seed its own gameplay RNG.

**Scope:** `game.observe(player_id)` supports all seats, with player 0
tracked incrementally and other seats reconstructed from public events.
Snapshots can be taken during dealing, bidding, and partially completed
tricks, but the current `Game.from_information_state()` rollout adapter
handles only `phase == "playing"` (card-decision points). Arbitrary-phase
continuations still require a general step-based engine. `sample_hands()`
guarantees feasible hidden hands: it uses exact uniform dealing for shared
candidate pools, rejection sampling for differing pools when successful,
and a potentially nonuniform matching fallback for restrictive cases.

Custom policies can override `export_policy_state()` and
`load_policy_state()` for their persistent JSON-compatible memory. If you
switch policy classes during an experiment, the new policy is responsible
for interpreting or discarding the old algorithm's memory.

```bash
python -m pytest -q
```


## Fast parallel Monte Carlo rollouts

`alphawhist.core.FastRollout` is a *history-free simulation backend* for
evaluating card decisions. The authoritative `Game` and immutable
`InformationState` still perform event validation and maintain public
observations, but no event log is updated in a rollout's inner loop.

```python
from alphawhist.core.FastRollout import evaluate_actions

observation = game.observe(player_id=0)
result = evaluate_actions(
    observation,
    n_rollouts=10000,
    seed=42,
    workers=8,
    objective="score",    # or "margin" for the Snake policy
)
print(result.best_action(), result.actions, result.mean_round_scores)
```

Features include 52-bit hand masks, integer card/trick comparisons, reuse of
the *same sampled hidden world* across candidate moves, deterministic
per-rollout RNG seeds (independent of worker count and batch size), and
`ProcessPoolExecutor` batch parallelism. Both `MonteCarloPlayer` and
`SnakeMonteCarloPlayer` use this evaluator for move selection in any player
seat. Player-specific private observations are reconstructed using only their
own hand and the public action history. Multi-process evaluation works with
the built-in `random` and `highest` rollout policies; custom callable
policies must be picklable if `workers > 1`.

Candidate sets sharing the same allowed-card pool use exact uniform dealing.
Constrained positions first attempt exact rejection sampling, then fall back
to randomized bipartite matching after 24 failures; this last resort is
*feasible but not guaranteed uniform*.

Use the bundled benchmark to measure real throughput on your hardware:

```bash
python -m benchmarks.rollouts --rollouts 100 --workers 1 --cards 7
python -m benchmarks.rollouts --rollouts 300 --workers 4 --cards 7
```

The benchmark compares full remaining-round simulations from the same
snapshot with the original object-based simulation. Process startup overhead
means multiple workers may be slower for small workloads. Search quality and
the statistical calibration of the rollout policy are separate issues.

## Monte Carlo card-score convergence

The rollout evaluator produces a separate expected **round point score for
each legal card**. To inspect how estimates and the chosen card change with
the rollout budget, run the seeded experiment:

```bash
python -m benchmarks.score_convergence --scenario early
python -m benchmarks.score_convergence --scenario mid --objective margin
python -m benchmarks.score_convergence --snapshot decision_42.json \\
    --rollouts 16 64 256 1024 4096 --reference 16384 --csv scores.csv
```

The report shows each candidate's estimated objective value, preferred card,
mean absolute deviation from a high-rollout reference and estimated regret
of the chosen card relative to that reference. It supports both `score`
(own expected round points) and `margin` (own points minus the largest
opponent **expected** points), and can export per-card CSV data.

All budgets share the same RNG seed and nested sampled worlds. The large
reference is also a finite Monte Carlo estimate, **not exact ground truth**.
Individual score estimates need not improve monotonically as N grows; the
regression tests check exact prefix accounting and aggregate convergence
against an analytically known one-card expectation.

## Default Monte Carlo budget

The default card-decision budget is **1,024 rollouts** in
`FastRollout.evaluate_actions()`, `MonteCarloPlayer`, and
`SnakeMonteCarloPlayer`. Both bot classes use this value even if
`additionalParameters` is absent or empty. The worker default remains one.

Explicit configuration overrides are unchanged; for example,
`{"monteCarloNumber": "128", "monteCarloWorkers": "2"}` still requests
128 rollouts with two workers. Existing numbered config presets (such as
`ThreePlayerMonte1000.json`) retain their named, explicit budgets.

`whistmain.py` still starts a game with random players unless its
player config is changed; setting the Monte Carlo default does not change
the default player lineup.
