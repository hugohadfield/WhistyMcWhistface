
# WhistyMcWhistface - A contract whist platform for AI research

WhistyMcWhistface is an implementation of the rules of contract whist along with useful functionality for writing bots to play the game.

The current state of the art on the platform SnakeMonteCarloPlayer performs a monte carlo simulation of future moves and plays to maximise the lead of the AI agent over the highest scoring other player in the game

Some limited GUI functionality is provided via the easygui library

TrumpPicker was written by Ed Dakin

# Getting started

The best entrypoint is whistmain.py which is set up to play several games of a given test config file



## Serializable information state and rollout experiments

The authoritative hard-constraint beliefs of **player 0** are now tracked
independently of other players' actual hands. Observations are immutable,
versioned, JSON-serializable and updated **only from public events**. Each
snapshot includes player 0's remaining hand, per-opponent candidate cards
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

**Scope:** `game.observe()` supports player 0; multi-observer beliefs would
need independently tracked information states. Snapshots can be taken
during dealing, bidding, and partially completed tricks, but the current
`Game.from_information_state()` rollout adapter handles only
`phase == "playing"` (card-decision points). Full arbitrary-phase
continuations will require the planned step-based game engine.
`sample_hands()` guarantees consistent hidden hands, but its randomized
matching does *not* guarantee uniform sampling over feasible deals.

Custom policies can override `export_policy_state()` and
`load_policy_state()` for their persistent JSON-compatible memory. If you
switch policy classes during an experiment, the new policy is responsible
for interpreting or discarding the old algorithm's memory.

```bash
python -m pytest -q
```
