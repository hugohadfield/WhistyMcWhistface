# Analytical two-trick endgame: experimental policy

New `AnalyticBidAware3Player` is an **opt-in**, policy-comparison variant.
It keeps `RandomPlayer` bidding and trump selection; it keeps the existing
unchanged `BidAware3Player` decisions except when it has exactly three
cards and at least two legal choices. At that point it evaluates each
candidate first card by an analytical three-trick approximation.

## 1. Finite-population void and follow-suit probabilities

The observer's `InformationState` contains public played cards, known void
suits, and *candidate* opponent cards, never opponents' real hands.

For each opponent with `m` cards remaining, `N` candidate cards,
and `S` candidates in the led suit, use

`P(void) = C(N-S, m) / C(N, m)`.

Under the **uniform feasible hand / uniformly random legal response**
model, the probability a particular follow-suit card gets played is
`(1-P(void))/S`. A particular off-suit card gets played with probability
`P(void)/(N-S)`. Only cards beating our candidate are accumulated.
Multiply each *future* opponent's marginal probability of **not**
beating our card; previously played pile cards are known exactly.

The hypergeometric expression is exact for **one opponent's marginal
under the stated model**, not necessarily a calibrated opponent policy.
Multiplying distinct opponents' probabilities assumes approximate
independence, since their unseen hands actually share a deck.

## 2. Joint probability over the final two tricks

For each candidate first three-card play, branch on winning or losing
the current trick. In the following two-card continuation, analytically
enumerate every feasible unordered opponent *two-card pair*, including
which card the opponent plays first under a random legal policy, its
remaining last card, mandatory follow-suit, trump strength, and who leads
the next trick.

This yields **joint** probabilities of winning zero, one or both of the
remaining two tricks. The 10-point exact-bid bonus is then evaluated
directly. The two possible own card orders are averaged with the same
softmax temperature as the existing heuristic.

In **two-player** games the two-trick distribution is exact under the
uniform opponent-pair and random-legal-first-card assumptions.
In **three-plus-player** games we compute one-on-one two-card marginals
and combine them as a cheap approximation; multiple opponents create
extra dependencies over who leads. In the initial three-to-two transition
we do **not** calculate a full outcome-conditioned posterior over
opponents' cards, only the lead/follow branch and reduced hand sizes.

Thus this is an analytical approximation, **not** hidden-hand
enumeration over all opponents, and **not** Monte Carlo. Complexity grows
with opponent possible-card pair counts, making it notably more
expensive than the original constant-time rank tables. Benchmark runtime
before adopting it for latency-sensitive applications.

## Experiments

Quick local targeted tests:

```bash
python -m pytest -q tests/test_analytic_endgame.py
```

Compare against the original three-card heuristic while matching dealt
hands and rotating the analytic player through all seats:

```bash
python -m benchmarks.bid_aware_3_tournament \
  --challenger analytic3 --baseline aware3 \
  --players 3 --games 600 --workers 4 \
  --seed 20261133 --json runs/analytic3-vs-aware3.json
```

Compare action valuations and empirical current-trick win calibration
on independently generated actor-safe traces against an independent
1,024-rollout MC reference:

```bash
python -m benchmarks.analytic_endgame_audit \
  --output-dir runs/analytic-audit \
  --games-per-player 80 --max-states 120 \
  --mc-rollouts 1024 --workers 4
```

Do not infer that better agreement with a Monte Carlo reference
automatically means stronger real match-winning play. The reference
uses `bid_aware_2` continuation and is not a ground truth oracle.
