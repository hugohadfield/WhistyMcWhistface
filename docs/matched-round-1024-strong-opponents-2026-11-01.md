# Strong-opponent matched-round A/B evaluation (1024 rollouts)

This experiment compares **actual round outcomes** between the current
`random` future-play Monte Carlo evaluator and `bid_aware_2`, using
the identical deterministic deal and tested seat for both replays.

## Why this follow-up?

The earlier [512-rollout matched-round experiment](paired-round-policy-benchmark-2026-11-01.md)
tested against two `RandomPlayer` opponents, with an inconclusive mean score
difference of -0.157 points per paired round (B minus A).
The full-match experiment, in contrast, contained two Monte Carlo competitors
and one random third player and observed +1.188 points per complete game.
Those are different environments and different performance metrics; the
experiments cannot be compared as if they had identical conditions.

This follow-up uses **1,024 rollouts for every Monte Carlo agent**, and
compares two opponents environments with the same independent deal seeds:

- `--opponents random`: two isolated-seed `RandomPlayer` opponents, to check
  whether the effect changes when the rollout budget doubles.
- `--opponents monte_carlo`: two fixed `bid_aware_2` Monte Carlo opponents,
  each running 1,024 rollouts. Only the *tested* seat changes policy between
  A and B; the opponents' algorithms remain fixed.

Both conditions use three players, 200 independently dealt scenarios for each
of the **3-, 4-, 5-, 6-, and 7-card** rounds (1,000 paired deals or 2,000 replayed
rounds *per opponent condition*). We omit 1 and 2 cards because the candidate
policy cannot affect a round that starts with 1 or 2 cards under the current
rollout evaluator: after trying a candidate play, any possible two-card
decision is already forced or absent.

Every scenario logs the tested player's score delta, margin delta, exact-bid
success, tricks won, and a fingerprint identifying the starting deal. The
reported 99% percentile-bootstrap intervals resample **paired deals**.
These are exploratory analyses, not a guarantee of positive improvement:
the reported overall mean weights each hand size equally, rather than using
the original 13-round match frequency distribution.

## Reproduction

```bash
python -m benchmarks.paired_round_policy \
  --rounds-per-size 200 --cards 3 4 5 6 7 \
  --rollouts 1024 --players 3 --workers 4 --seed 20262817 \
  --opponents monte_carlo --csv paired-monte-carlo-1024.csv

python -m benchmarks.paired_round_policy \
  --rounds-per-size 200 --cards 3 4 5 6 7 \
  --rollouts 1024 --players 3 --workers 4 --seed 20262817 \
  --opponents random --csv paired-random-1024.csv
```

Both datasets run in separate GitHub Actions matrix jobs; each job has a
30-minute timeout and uploads the CSV with `if: always()`, even after a
failure or cancellation. The script flushes each completed scenario as it
arrives, though a catastrophic worker shutdown can still interrupt upload.

## What this tests, and what it does not

This is a paired *single-seat intervention* against a fixed opponent field.
We preserve dealer, trump, bids, hands, player order, and the opponents'
PRNG streams at the start of a round. We do **not** force opponents to play
identical cards after the trajectories diverge; those differences are a
legitimate consequence of changing policy. All agents see only their own
cards and public game history, not opponents' sampled real hands.

It measures per-round expected-score playing strength, not probability of
winning a 13-round match. It tests only three-player play and a fixed
opponent policy; more robust conclusions require several opponent mixes,
player counts, independently held-out deals, and full-game confirmation.
