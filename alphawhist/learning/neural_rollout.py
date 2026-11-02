"""Information-safe, batched rollouts with neural play for the observer only.

The observer's future cards are chosen by a frozen DistributionalWhistMLP.
Other simulated players keep a specified cheap continuation policy. Crucially,
the model sees only the initial observer InformationState advanced by PUBLIC
events, never sampled opponents' hands. All full hands stay inside SimState.

The same hidden world and continuation RNG seed are reused across candidate
actions, as in FastRollout. Each checkpoint is loaded once per worker process.
No nested Monte Carlo search is performed during continuation.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random

from alphawhist.core.FastRollout import (
    CARD_ID, CARDS, Evaluation, SimState, _choose_card, _mix, card_ids,
    hand_mask, legal_mask,
)
from alphawhist.core.InformationState import InformationState

_OPPONENT_POLICIES = ("random", "highest", "bid_aware_2")


@dataclass
class _Branch:
    state: SimState
    observation: InformationState
    rng: random.Random
    action_index: int


@lru_cache(maxsize=2)
def _load_model(checkpoint):
    # Importing PyTorch is optional for the rest of the Whist engine.
    import torch
    from alphawhist.learning.checkpoint import load_checkpoint

    # Each exporter worker is a separate process. Small CPU inference batches
    # are substantially faster without OpenMP oversubscription.
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint, device="cpu")
    return model


def _advance(branch, card):
    """Apply the same PUBLIC card and trick events seen by the real observer."""
    state = branch.state
    if not legal_mask(state.hands[state.turn], state.pile) & (1 << card):
        raise ValueError("Continuation selected an illegal card")
    actor = state.turn
    observation = branch.observation.apply_public_event(
        {"type": "card", "player": actor, "card": CARDS[card]})
    state._play_legal(card)
    if not state.pile:
        observation = observation.apply_public_event(
            {"type": "trick_end", "winner": state.leader})
    if observation.turn != state.turn or observation.leader != state.leader:
        raise AssertionError("Public observation diverged from simulated game")
    branch.observation = observation


def _until_neural_turn(branch):
    """Advance opponents/forced plays until a nonforced observer decision."""
    state = branch.state
    me = branch.observation.observer_id
    while state.pile or any(state.hands):
        legal = legal_mask(state.hands[state.turn], state.pile)
        if not legal:
            raise AssertionError("Unfinished simulation has no legal card")
        if state.turn == me:
            if legal.bit_count() > 1:
                return True
            choice = next(card_ids(legal))
        else:
            choice = _choose_card(state, legal, branch.rng, branch._opponent_policy)
        _advance(branch, choice)
    return False


def _finish_group(branches, model, opponent_policy, sums, counts, previous_won):
    """Batch all observer network calls across simultaneously active branches."""
    from alphawhist.learning.encoding import encode_batch

    for branch in branches:
        branch._opponent_policy = opponent_policy

    live = branches
    while live:
        waiting = []
        for branch in live:
            if _until_neural_turn(branch):
                waiting.append(branch)
            else:
                i = branch.action_index
                for j, score in enumerate(branch.state.round_scores()):
                    sums[i][j] += score
                if counts is not None:
                    additional = (branch.state.tricks_won[branch.observation.observer_id]
                                  - previous_won)
                    if not 0 <= additional <= 7:
                        raise AssertionError("Invalid observer trick histogram")
                    counts[i][additional] += 1
        if not waiting:
            break
        encoded = encode_batch([b.observation for b in waiting])
        predictions = model.predict_encoded(encoded)
        selected = predictions.best_card_ids().cpu().tolist()
        for index, (branch, card) in enumerate(zip(waiting, selected)):
            if not encoded.legal_mask[index, card]:
                raise AssertionError("Neural continuation proposed an illegal card")
            if not legal_mask(branch.state.hands[branch.state.turn],
                              branch.state.pile) & (1 << card):
                raise AssertionError("Neural action disagrees with sampled simulation")
            _advance(branch, card)
        live = waiting


def _neural_batch(args, collect_histograms=False):
    (snapshot, actions, start, stop, seed, checkpoint, opponent_policy,
     worlds_per_batch) = args
    model = _load_model(checkpoint)
    own = snapshot.observer_id
    action_ids = [CARD_ID[card] for card in actions]
    sums = [[0] * snapshot.num_players for _ in actions]
    counts = [[0] * 8 for _ in actions] if collect_histograms else None
    deal_rng = random.Random()
    previous_won = snapshot.tricks_won[own]

    for first in range(start, stop, worlds_per_batch):
        branches = []
        for index in range(first, min(first + worlds_per_batch, stop)):
            deal_rng.seed(_mix(seed, 2 * index))
            hands = snapshot.sample_hands(deal_rng)
            base = SimState.from_snapshot(snapshot, hands)
            policy_seed = _mix(seed, 2 * index + 1)
            for action_index, action in enumerate(action_ids):
                branch = _Branch(
                    state=SimState(base.hands.copy(), base.pile.copy(),
                                   base.bids, base.tricks_won.copy(),
                                   base.leader, base.turn, base.trump),
                    observation=snapshot,
                    rng=random.Random(policy_seed),
                    action_index=action_index,
                )
                _advance(branch, action)
                branches.append(branch)
        _finish_group(branches, model, opponent_policy, sums, counts, previous_won)
    return sums, counts


def evaluate_neural_self(snapshot, actions=None, *, checkpoint,
                         n_rollouts=128, seed=0, workers=1, batch_size=128,
                         worlds_per_batch=16, opponent_policy="bid_aware_2",
                         collect_trick_histograms=False):
    """Evaluate actions with the frozen net controlling OUR future turns only.

    Opponents continue according to opponent_policy, without nested search.
    Returns the standard FastRollout.Evaluation object, including optional
    eight-bin *additional* observer trick counts for supervised training.
    """
    if not isinstance(snapshot, InformationState):
        raise TypeError("A real observer InformationState is required")
    if snapshot.phase != "playing" or snapshot.turn != snapshot.observer_id:
        raise ValueError("Snapshot must be at the observer's card decision")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("A frozen, accessible neural checkpoint is required")
    if opponent_policy not in _OPPONENT_POLICIES:
        raise ValueError("Unknown opponent continuation policy")
    if min(n_rollouts, workers, batch_size, worlds_per_batch) < 1:
        raise ValueError("Rollout count and batch sizes must be positive")

    legal = legal_mask(hand_mask(snapshot.own_hand),
                       [CARD_ID[card] for _, card in snapshot.pile])
    actions = (tuple(CARDS[i] for i in card_ids(legal))
               if actions is None else tuple(actions))
    if not actions or len(actions) != len(set(actions)):
        raise ValueError("Actions must be distinct and nonempty")
    if any(card not in CARD_ID or not legal & (1 << CARD_ID[card])
           for card in actions):
        raise ValueError("All candidate actions must be legal observer cards")

    checkpoint = str(Path(checkpoint).resolve())
    tasks = [(snapshot, actions, begin, min(begin + batch_size, n_rollouts),
              seed, checkpoint, opponent_policy, worlds_per_batch)
             for begin in range(0, n_rollouts, batch_size)]
    totals = [[0] * snapshot.num_players for _ in actions]
    counts = [[0] * 8 for _ in actions] if collect_trick_histograms else None

    def accumulate(part):
        score_sums, hist = part
        for i, row in enumerate(score_sums):
            for j, value in enumerate(row):
                totals[i][j] += value
        if counts is not None:
            for i, row in enumerate(hist):
                for j, value in enumerate(row):
                    counts[i][j] += value

    kernel = _neural_batch_hist if collect_trick_histograms else _neural_batch
    if workers == 1:
        for task in tasks:
            accumulate(kernel(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(kernel, tasks):
                accumulate(part)
    if counts is not None and any(sum(row) != n_rollouts for row in counts):
        raise AssertionError("Incomplete neural continuation histogram")

    means = tuple(tuple(score / n_rollouts for score in row) for row in totals)
    return Evaluation(
        tuple(actions), means,
        tuple(row[snapshot.observer_id] for row in means),
        n_rollouts,
        None if counts is None else tuple(tuple(row) for row in counts),
    )


def _neural_batch_hist(args):
    return _neural_batch(args, collect_histograms=True)
