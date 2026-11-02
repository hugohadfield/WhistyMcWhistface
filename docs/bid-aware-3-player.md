# Bid-aware three-card policy

`BidAware3Player` extends `BidAware2Player`. Both inherit **exactly the
same bidding and trump selection as RandomPlayer**. At 3 cards remaining
with at least 2 legal options, it uses a constant-cost 3-card lookahead.
At 2 cards it calls the existing **unchanged** `bid_aware_2` policy;
at 4+ cards it chooses randomly.

For each legal first card `a`, an approximate chance of taking the
current trick is derived from card rank, trump, opponents yet to play,
and the **public** pile, using a new three-card probability lookup table.

The resulting score is

```text
Q3(a) = p(win now | a) * (1 + V2(other two, needed-1, leading next))
      + p(lose now | a) * V2(other two, needed, following next)
```

`V2` is **not a max over the next two-card moves**: it integrates
the two possible next-card orders with the actual stochastic softmax
temperature of bid-aware-2. It estimates last-trick win/loss probabilities
and the **10-point bonus** for exactly hitting the bid. The three-card
move is then chosen with the same softmax temperature.

No nested Monte Carlo, no opponent hidden-hand access, and no inference
from privileged game state. The predicted future pile and follow-suit
constraints are approximated, **not** fully searched. The probabilities
are hand-designed and are not calibrated to particular opponents.

## Matched-deal tests

To measure **incremental gain** over existing bid-aware-2, run:

```bash
python -m benchmarks.bid_aware_3_tournament \
  --baseline aware2 --players 3 --games 600 --workers 4 \
  --seed 20261125 --json runs/bid-aware3-vs-aware2-3p.json
```

To measure performance against fully random card play:

```bash
python -m benchmarks.bid_aware_3_tournament \
  --baseline random --players 3 --games 600 --workers 4 \
  --seed 20261127 --json runs/bid-aware3-vs-random-3p.json
```

The same commands work with `--players 2 --games 400`. For 4–7 players,
set a game count that is a multiple of the player count, so the
challenger rotates through all seats on each deal.

Tournament output includes full-game first-place share (ties split),
deal-clustered bootstrap 95% intervals, score margin, exact-bid rates,
overtrick/undertrick rates, and counts of nonforced three-card decisions.
In 3-player tournaments, **neutral first-place share is 1/3**, not 1/2.

The policy can also be used as a cheap continuation in normal MC:

```python
MonteCarloPlayer(0, additionalParameters={"monteCarloPolicy": "bid_aware_3"})
```

The original policies `random` and `bid_aware_2` are unchanged.
