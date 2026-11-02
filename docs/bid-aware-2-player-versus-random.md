# Standalone bid_aware_2 vs RandomPlayer

`bid_aware_2` was previously a **cheap rollout policy**, not a
standalone `Player`. `BidAware2Player` exposes it without Monte Carlo:
it inherits `RandomPlayer`, changing **only** decisions with exactly two
cards in hand and two legal cards to the existing bid-aware softmax
heuristic in `alphawhist.core.EndgamePolicy.pick_bid_aware_two_card`.

All other plays remain uniformly random among legal choices. Both agents
use exactly the same `RandomPlayer` bidder (roughly cards/players,
adjusted for legal restrictions) and trump picker. This is **not** the
`MonteCarloBidPlayer` that runs searches over hypothetical bids.

The real player uses the actor's private hand plus only public
bids, trick counts, trump, and the current trick. It constructs the same
minimal continuation state as the rollout player, leaving all opponent
hand masks zero. No hidden opponents' cards are read.

## Compare against random, controlling for the initial deal and seats

Two-player head-to-head (200 independently dealt full matches, each
played twice with the opponent seats exchanged):

```bash
python -m benchmarks.bid_aware_2_vs_random \
  --players 2 --games 400 --workers 4 \
  --seed 20261117 --progress-every 40 \
  --json runs/bid-aware-2-vs-random-2p.json \
  --csv runs/bid-aware-2-vs-random-2p.csv
```

Three-player baseline (200 independently dealt full matches, each
played three times with the bid-aware player in each seat):

```bash
python -m benchmarks.bid_aware_2_vs_random \
  --players 3 --games 600 --workers 4 \
  --seed 20261116 --progress-every 60 \
  --json runs/bid-aware-2-vs-random-3p.json \
  --csv runs/bid-aware-2-vs-random-3p.csv
```

Outputs include tied-first-place credit, full-game win share, score margin
over the strongest random opponent, exact-bid rate, overtrick/undertrick
rate, and *deal-clustered* bootstrap confidence intervals. Under no
card-play advantage, first-place share is 50% in 2-player games and 33.3%
in 3-player games, not 50% in both.

Don't confuse the bid-aware two-card continuation with a full strategic
player: the other ~90% of non-forced decisions are selected randomly.
The heuristic is stochastic (softmax temperature 1.5 points), so even
in two-card endgames it does not always select the highest-rated card.
