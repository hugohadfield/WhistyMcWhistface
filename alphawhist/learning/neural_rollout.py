"""Information-safe, batched neural continuation for self and (optionally) opponents.

The acting neural policy sees only its own sampled hand and the public card
history. Sampled other players' hands never enter the actor's features.
Heuristic-opponent mode remains unchanged. Opponent policy `neural` uses the
same frozen model in ALL simulated seats, each with independently reconstructed
private knowledge. Candidate actions share sampled worlds and continuation RNG.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import random

from alphawhist.core.FastRollout import (
    CARD_ID, CARDS, Evaluation, SimState, _choose_card, _mix, card_ids,
    hand_mask, legal_mask,
)
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.fast_observation import FastObserver, encode_fast_batch

_OPPONENT_POLICIES = ("random", "highest", "bid_aware_2", "neural")


@dataclass
class _Branch:
    state: SimState
    observation: InformationState | FastObserver
    rng: random.Random
    action_index: int
    all_observations: tuple[FastObserver, ...] | None = None


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
    observation = branch.observation
    if branch.all_observations is not None:
        # All neural actors observe the same PUBLIC play, each in their own
        # perspective. No actor reads another sampled private hand.
        for pov in branch.all_observations:
            pov.play_card(actor, card)
        state._play_legal(card)
        if not state.pile:
            for pov in branch.all_observations:
                pov.end_trick(state.leader)
        if any(pov.turn != state.turn or pov.leader != state.leader
               for pov in branch.all_observations):
            raise AssertionError("Neural perspectives diverged from public state")
    elif isinstance(observation, FastObserver):
        observation.play_card(actor, card)
        state._play_legal(card)
        if not state.pile:
            observation.end_trick(state.leader)
    else:
        # Slow reference path retained for differential correctness tests.
        observation = observation.apply_public_event(
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
    all_neural = branch.all_observations is not None
    while state.pile or any(state.hands):
        legal = legal_mask(state.hands[state.turn], state.pile)
        if not legal:
            raise AssertionError("Unfinished simulation has no legal card")
        if state.turn == me or all_neural:
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
        observations = [
            (b.all_observations[b.state.turn] if b.all_observations is not None
             else b.observation)
            for b in waiting
        ]
        encoded = (encode_fast_batch(observations)
                   if isinstance(observations[0], FastObserver)
                   else encode_batch(observations))
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
    observer_template = FastObserver.from_snapshot(snapshot)
    # Parse public round history ONCE per batch. Last round_start excludes
    # earlier-round events (including previously played duplicate card IDs).
    round_events = None
    if opponent_policy == "neural":
        events = [json.loads(raw) for raw in snapshot.history_json]
        start = next((i for i in range(len(events) - 1, -1, -1)
                      if events[i]["type"] == "round_start"), None)
        if start is None:
            raise ValueError("All-neural rollout needs a public round history")
        round_events = tuple(events[start:])

    for first in range(start, stop, worlds_per_batch):
        branches = []
        for index in range(first, min(first + worlds_per_batch, stop)):
            deal_rng.seed(_mix(seed, 2 * index))
            hands = snapshot.sample_hands(deal_rng)
            base = SimState.from_snapshot(snapshot, hands)
            policy_seed = _mix(seed, 2 * index + 1)
            all_templates = None
            if opponent_policy == "neural":
                # Each perspective gets its OWN sampled remaining cards only.
                # The other sampled hands never enter this perspective.
                all_templates = tuple(
                    observer_template if pid == own else
                    FastObserver.from_sampled_perspective(
                        snapshot, pid, hands[pid], round_events)
                    for pid in range(snapshot.num_players)
                )
            for action_index, action in enumerate(action_ids):
                observers = (tuple(pov.fork() for pov in all_templates)
                             if all_templates is not None else None)
                branch = _Branch(
                    state=SimState(base.hands.copy(), base.pile.copy(),
                                   base.bids, base.tricks_won.copy(),
                                   base.leader, base.turn, base.trump),
                    observation=(observers[own] if observers is not None
                                 else observer_template.fork()),
                    rng=random.Random(policy_seed),
                    action_index=action_index,
                    all_observations=observers,
                )
                _advance(branch, action)
                branches.append(branch)
        _finish_group(branches, model, opponent_policy, sums, counts, previous_won)
    return sums, counts


def evaluate_neural_self(snapshot, actions=None, *, checkpoint,
                         n_rollouts=128, seed=0, workers=1, batch_size=128,
                         worlds_per_batch=16, opponent_policy="bid_aware_2",
                         collect_trick_histograms=False):
    """Evaluate actions using frozen neural future play for our own turns.

    Set opponent_policy="neural" for same-model opponents, each with only
    their own sampled hand and public knowledge. Otherwise opponents use
    the specified cheap heuristic. No nested search is performed.
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
