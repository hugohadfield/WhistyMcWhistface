# Neural state encoder and distributional MLP (V0)

This is **infrastructure, not a trained agent**. The production Monte Carlo
player and all its defaults remain unchanged. There is no checkpoint and
no neural rollout policy wired into `FastRollout` yet.

## Inputs and information boundary

Use `alphawhist.learning.encoding.encode_decision(observation)` with a
`Game.observe(player_id)` snapshot while that player is about to act.
The function **does not accept Game or SimState** because those can
contain real hidden hands. It uses `InformationState`, which stores only:

- The observer's actual hand, valid plays, known public played cards and pile.
- Trump and lead suit; public current-round card ownership history.
- Bids, won tricks, relative seat positions, match scores.
- Each opponent's *feasible* cards and publicly deduced void suits.

Feasible ownership is NOT actual ownership, and no target label is used
when building features. All previous private policy memory is ignored.

Card IDs match `FastRollout.CARD_ID`: suit order `h,c,s,d`, ranks 2..14.
Player slots are relative to the current observer (self at 0, then the
next players in clockwise turn order). Slots 0..6 allow 2..7 players.
Suit roles (trump and lead) are explicit, but this V0 does not enforce
suit-permutation equivariance.

### Layout (feature schema version 1)

| Tensor | Shape | Channels |
|---|---:|---|
| Cards | 52 × 27 | Own, played, in pile, legal, trump suit, lead suit, plus seven relative-player slots each for public played-by, pile-by, feasible-for |
| Players | 7 × 16 | Active, self, dealer, leader, turn, normalized bid, tricks won, bid deficit, cards remaining, score, score difference, four void-suit flags, currently in pile |
| Global | 14 | Normalized player/round/trick/pile counts and one-hot trump/lead suits |

The encoder concatenates these to a **1,530-element float32 vector**
(52 × 27 + 7 × 16 + 14). A separate 52-element boolean legal-action
mask accompanies the vector. Further metadata includes remaining tricks,
won tricks and bid for exact scoring.

Only playing-phase decisions with the observer to act are accepted.
`encode_batch(states)` returns stacked NumPy arrays suitable for
`torch.as_tensor`.

## Neural network

`alphawhist.learning.model.DistributionalWhistMLP` is optional PyTorch:

- Linear(1530,128), LayerNorm, SiLU.
- Two residual blocks: LayerNorm, Linear, SiLU, Linear with residual skip.
- Linear(128,416), reshaped to **[batch, 52 cards, 8 outcomes]**.

For each candidate card the categorical distribution describes 0..7
**additional tricks won by the acting player**, not total points.
`model.predict(...)` masks impossible additional trick counts and all
illegal plays and uses the exact score rule

```text
Q(card) = tricks_won + E[additional_tricks | card]
          + 10 * P(additional_tricks == bid - tricks_won | card)
```

Thus the bid bonus is not a separate arbitrary neural regression head.
`ActionPredictions.best_card_ids()` returns a batched greedy legal
card choice. For supervised learning use `distributional_loss` on
per-card future-win histograms from Monte Carlo rollouts. Forced actions
still respect the legality mask.

The default architecture is only **316,448 trainable parameters**.
The true parameter count can be inspected with
`sum(p.numel() for p in model.parameters())`.

## Installation and use

```bash
# CPU-only environment (optional; the standard bot does not require torch):
python -m pip install -r requirements-dev.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pytest -q tests/test_learning_encoding.py tests/test_learning_model.py
```

Alternatively `pip install -r requirements-ml.txt` installs the
optional model dependency through the default configured PyPI index.

```python
from alphawhist.learning.encoding import encode_decision
from alphawhist.learning.model import DistributionalWhistMLP

# During a real game, after trump selection and bidding:
# snapshot = game.observe(player_id)
encoded = encode_decision(snapshot)

model = DistributionalWhistMLP(hidden_size=128, residual_blocks=2)
predictions = model.predict_encoded(encoded)
card_id = predictions.best_card_ids()[0].item()
# alphawhist.learning.encoding.CARDS[card_id] is the chosen legal card.
```

The model above has **random initial weights**. Do not play it as if it
were a trained policy.

## Next step: training data

For each frozen observer snapshot and every legal card, sample plausible
hidden worlds as `FastRollout.evaluate_actions` does, run a specified
future-play policy, and **count additional tricks won by the observer**
for each action. Normalize each histogram to eight bins and mask
impossible counts. Using different target continuation policies produces
different value functions.

Split by original *deals or matches* rather than random per-decision rows
to prevent game-history leakage between training and test. Keep the
sampling seeds and the future-play policy version in dataset metadata.
Avoid training on the adversarial simulator's real opponents' cards: only
the label-generation process may see sampled hidden worlds.

After training, compare held-out prediction accuracy, legal-card rankings,
actual score on matched deals, and full-match strength against frozen
baseline policies. Integration into `finish_round` should only occur
after measuring CPU inference overhead; per-card PyTorch calls inside
thousands of individual rollouts would likely be too expensive without
batching or leaf evaluation.

## Model and schema versioning

The encoder exposes `FEATURE_SCHEMA_VERSION = 1`. Persist this value
alongside the model hidden width, residual depth, dataset generation
parameters and training policy. Do not silently load a checkpoint with
a different feature layout.
