# Paired single-round policy A/B benchmark

This experiment measures the **actual round-score effect** of changing the
Monte Carlo future-play policy, rather than agreement with another heuristic
or the outcome of a complete 13-round tournament.

## Design

For each independent deal seed and hand size:

1. Draw one legal set of hands (one to seven cards per player), choose a
   dealer, and rotate the evaluated seat across scenarios.
2. Run a fresh round with Monte Carlo continuation policy A in that seat.
3. Reset to the **same player hands, seat, dealer and opponent RNG streams**
   and rerun with continuation policy B. Everything else is unchanged.
4. Assert that chosen trumps and every player's bids match across replays.
   (All trump/bid methods in this experiment are deterministic functions
   of the same hands and player state.)
5. Store the tested player's actual score difference, exact-bid indicators,
   won tricks, and score margin against the highest-scoring opponent.

Opponents are seeded RandomPlayer variants with **independent per-player
random streams**. The tested MonteCarloPlayer receives a deterministic seed
per public decision index and has equal rollout budgets in both replays.

The tested player **never receives the opponents' actual hands**, and
the information-state path is the real Game observer API. The test harness
does know the full deal, because it needs to replay the exact same round.
Different policies may cause different future game states and opponent
legal moves; matching starts and RNG streams does *not* force identical
future events.

The one-card round is a useful negative control: no card choice can differ,
so the two variants should receive identical scores on each deal.

## Running

Small local smoke test:

    python -m benchmarks.paired_round_policy --rounds-per-size 5 \
        --cards 1 2 3 4 5 6 7 --rollouts 16 --workers 1

Regular experiment:

    python -m benchmarks.paired_round_policy --rounds-per-size 1000 \
        --cards 1 2 3 4 5 6 7 --rollouts 512 --workers 4 \
        --seed 20261211 --csv matched-rounds.csv

For larger player counts:

    python -m benchmarks.paired_round_policy --rounds-per-size 100 \
        --cards 2 3 4 5 6 7 --players 7 --rollouts 64 --workers 4 \
        --csv seven-players.csv

The PR's experimental CI job runs **100 independently paired rounds at
each hand size**, a total of 700 paired scenarios / 1,400 round replays,
with 512 Monte Carlo rollouts for both policies.

CSV is written and **flushed after each completed pair**, so partial
results can be recovered on a timeout. The CI upload step uses
'if: always()' and retains any existing CSV even after a benchmark failure.
GitHub can still fail to save a file if the runner crashes before artifact
upload; this is not absolute fault tolerance.

## Interpretation

The primary A/B observation is the **paired score delta**

    D = score_B - score_A

A positive result favors the bid-aware continuation policy.
Additional metrics are the paired margin against the strongest opponent
and the change in exact-bid success probability.

The summary prints a **99% percentile bootstrap confidence interval** for
the mean paired score delta, with each independently dealt round as the
resampling unit. It prints one overall estimate and separate diagnostics
for each hand size. The separate 99% per-size intervals are descriptive
and **not** corrected for testing multiple hand sizes.

The overall estimate uses an equal-weight mixture of the requested hand
sizes, which is a useful development benchmark but is **not** the original
13-round schedule's mixture of sizes (7,6,5,4,3,2,1,2,3,4,5,6,7).
If optimizing for that schedule, compute a prespecified weighted estimate
or sample according to its frequencies.

This is a 3-player-versus-random-opponents experiment by default,
not proof of performance against expert opponents. Repeated tuning on
these seeds will overfit the benchmark. For a promotion decision,
freeze the candidate and run *fresh, held-out seeds* with a predeclared
primary statistic and sample size.

## Reproducibility

The CSV includes the deal seed, tested seat, dealer, hand-size, all bids,
trump, and SHA-256 fingerprint of all starting hands. It does **not**
include the hidden cards in the CSV. The deal is fully reconstructible
using the committed code, seed and Python RNG. Across worker counts,
the outcome fields are deterministic; elapsed seconds are not.
