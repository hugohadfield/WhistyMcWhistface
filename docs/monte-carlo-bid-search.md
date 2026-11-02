# Monte Carlo bidding with cheap simulated card play

`MonteCarloBidPlayer` is an **opt-in** alternative to `MonteCarloPlayer`.
It keeps the same card-playing search, trump selection, and game rules.
Only the method for choosing the bid changes.

At the moment of bidding, the game passes the bidder's own
`InformationState` (own hand, trump, current game scores, and public
history). For **every legal bid**, the bidder:

1. Samples the other players' hidden hands from the observer's feasible
   hidden-hand distribution. It never reads actual opponents' hands.
2. Preserves earlier publicly announced bids and assigns still-unannounced
   bids using the cheap average-bid heuristic. This respects the dealer
   no-exact-sum and two-consecutive-zero restrictions.
3. Plays out the whole round using the `bid_aware_2` continuation,
   which makes random plays with a bid-aware two-card endgame.
4. Scores its hypothetical round, and repeats for the configured rollout
   budget. Each bid uses the **same sampled worlds and continuation seeds**
   to reduce variance in comparisons.
5. Selects the legal bid with maximum expected own round score.

```python
from MonteCarloBidPlayer import MonteCarloBidPlayer

player = MonteCarloBidPlayer(0, additionalParameters={
    "monteCarloNumber": "1024",  # unchanged card-play search budget
    "monteCarloPolicy": "bid_aware_2",
    "bidRollouts": "1024",       # complete-round simulations per legal bid
})
```

## Paired complete-game tournament

Both competing bots use exactly the same `bid_aware_2` card-play policy and
rollout count. The third player is random. Each deal seed is played twice,
swapping the bidder and control player between seats:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.mc_bid_tournament \
  --games 100 --workers 4 --seed 20261109 \
  --play-rollouts 1024 --bid-rollouts 1024 \
  --json runs/mc-bid-vs-original.json \
  --csv runs/mc-bid-vs-original.csv
```

This prints running scores, paired wins, first-place win shares and
a paired-deal bootstrap interval. The original MC bidder remains the baseline.

## Important limitations

- This is **not** nested Monte Carlo. The bidding simulator does **not** run
  the expensive 1,024-rollout card decision search inside each simulated play.
  Its continuation is intentionally cheap, and need not match how the
  real MC agent eventually plays.
- `bid_aware_2` is mostly uniform random play until the two-card endgame;
  the simulated bid can have a relatively small impact until that stage.
- Before all opponents have bid, their future bids are **estimated using
  a simple average-bid heuristic**, not predicted from their sampled hands.
  The model does not use the bids already observed to reweight hidden deals.
- Optimizes **expected own round points**, not the chance of winning the
  complete 13-round match. Both are worth testing separately.
- This change makes no claim of improved playing strength without
  a sufficiently large, independently seeded matched-deal tournament.
