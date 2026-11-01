"""Opt-in, append-only, information-safe training traces from real Game play.

One JSONL record per decision, written after the round is scored so actual
outcomes are attached. The observation is EXACTLY what the acting player saw
before the move. The file contains no unseen/real opponent hands. An optional
separate privileged replay file may store real deals for controlled debugging;
NEVER feed that replay file into the model or teacher feature encoder.

The public-history field is clipped to the current round to avoid multiplying
earlier rounds' event histories by every decision. InformationState.from_dict
can restore the result without a source Game.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from alphawhist.core.InformationState import InformationState
from alphawhist.core.WhistLib import tricksToScore

TRACE_SCHEMA_VERSION = 1


def _emit(stream, record):
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()  # Retain completed rounds even if a later game crashes.


def _trim_snapshot(snapshot):
    """Remove policy memory and previous-round events before serializing."""
    if not isinstance(snapshot, InformationState):
        raise TypeError("Expected observer-specific InformationState")
    start = max(i for i, event in enumerate(snapshot.history_json)
                if json.loads(event)["type"] == "round_start")
    return replace(snapshot, history_json=snapshot.history_json[start:],
                   policy_state_json=None).to_dict()


class TrainingRecorder:
    """Context-managed trace writer; independent of PyTorch.

    Attach to `game.training_recorder` after constructing/resetting Game.
    No logging or observer reconstruction occurs when it is absent.

    Never use the optional privileged replay path as a training source.
    """

    def __init__(self, trace_path, game_id, metadata=None, privileged_path=None):
        self.trace_path = Path(trace_path)
        self.game_id = str(game_id)
        self.metadata = dict(metadata or {})
        self.privileged_path = Path(privileged_path) if privileged_path else None
        self._trace = None
        self._private = None
        self._pending = []
        self._active_round = None
        self.decisions_written = 0

    def __enter__(self):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace = self.trace_path.open("w", encoding="utf-8", newline="\n")
        if self.privileged_path is not None:
            if self.privileged_path.resolve() == self.trace_path.resolve():
                raise ValueError("Privileged replay and public trace must differ")
            self.privileged_path.parent.mkdir(parents=True, exist_ok=True)
            self._private = self.privileged_path.open("w", encoding="utf-8", newline="\n")
        _emit(self._trace, {
            "type": "game_start", "trace_schema": TRACE_SCHEMA_VERSION,
            "game_id": self.game_id, "metadata": self.metadata,
        })
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._private:
                self._private.close()
        finally:
            if self._trace:
                self._trace.close()
            self._private = self._trace = None

    def _require_open(self):
        if self._trace is None:
            raise RuntimeError("TrainingRecorder must be used as a context manager")

    def on_round_start(self, game):
        self._require_open()
        if self._pending:
            raise RuntimeError("Previous round has unflushed decisions")
        self._active_round = game.roundnumber
        # A digest makes the initial deal identifiable without exporting it.
        # The full hands are explicitly opt-in in a separate privileged file.
        hands = [list(p.real_hand) for p in game.players]
        digest = hashlib.sha256(
            json.dumps(hands, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _emit(self._trace, {
            "type": "round_start", "trace_schema": TRACE_SCHEMA_VERSION,
            "game_id": self.game_id, "round": game.roundnumber,
            "cards_per_player": game.cardnumbers[game.roundnumber],
            "dealer": game.dealer, "deal_sha256": digest,
            "players": game.numberofplayers,
        })
        if self._private is not None:
            _emit(self._private, {
                "type": "privileged_initial_hands",
                "game_id": self.game_id, "round": game.roundnumber,
                "hands_by_seat": hands, "deal_sha256": digest,
            })

    def on_decision(self, game, snapshot, legal, chosen, policy_name=None):
        self._require_open()
        if snapshot.phase != "playing" or snapshot.turn != snapshot.observer_id:
            raise ValueError("Decision snapshot must be at the acting observer")
        if game.roundnumber != self._active_round:
            raise ValueError("Decision belongs to a different round")
        if chosen not in legal or len(legal) != len(set(legal)):
            raise ValueError("Chosen action must be one of the distinct legal cards")
        if set(legal) - set(snapshot.own_hand):
            raise ValueError("Legal cards must be known to the acting player")
        self._pending.append({
            "type": "decision", "trace_schema": TRACE_SCHEMA_VERSION,
            "game_id": self.game_id,
            "deal_group": str(self.metadata.get("deal_group", self.game_id)),
            "round": game.roundnumber,
            "decision_in_round": len(self._pending),
            "player": snapshot.observer_id,
            "observation": _trim_snapshot(snapshot),
            "legal_cards": list(legal),
            "chosen_card": chosen,
            "behavior_policy": policy_name or type(game.players[snapshot.observer_id]).__name__,
            "behavior_rollouts": getattr(game.players[snapshot.observer_id],
                                         "monteCarloNumber", None),
            "behavior_continuation": getattr(game.players[snapshot.observer_id],
                                             "monteCarloPolicy", None),
        })

    def on_round_end(self, game):
        self._require_open()
        if self._active_round != game.roundnumber:
            raise ValueError("Round ending without a matching start")
        cards_per_player = game.cardnumbers[game.roundnumber]
        if len(self._pending) != game.numberofplayers * cards_per_player:
            raise ValueError("Missing card decisions in completed round")
        expected = [
            tricksToScore(bid, won)
            for bid, won in zip(game.bids, game.tricksWon)
        ]
        for record in self._pending:
            p = record["player"]
            won = game.tricksWon[p]
            previously_won = record["observation"]["tricks_won"][p]
            record["actual_remaining_tricks_won"] = won - previously_won
            record["final_tricks_won"] = won
            record["actual_round_score"] = expected[p]
            record["exact_bid"] = int(won == game.bids[p])
            record["final_round_scores"] = expected
            record["final_match_scores"] = list(game.scores)
            _emit(self._trace, record)
            self.decisions_written += 1
        _emit(self._trace, {
            "type": "round_end", "trace_schema": TRACE_SCHEMA_VERSION,
            "game_id": self.game_id, "round": game.roundnumber,
            "bids": list(game.bids), "tricks_won": list(game.tricksWon),
            "round_scores": expected, "match_scores": list(game.scores),
        })
        self._pending.clear()
        self._active_round = None

    def on_game_end(self, game):
        self._require_open()
        if self._pending:
            raise ValueError("Cannot finalize match with pending decisions")
        best = max(game.scores)
        _emit(self._trace, {
            "type": "game_end", "trace_schema": TRACE_SCHEMA_VERSION,
            "game_id": self.game_id,
            "match_scores": list(game.scores),
            "winning_seats": [i for i, score in enumerate(game.scores)
                              if score == best],
            "decisions_written": self.decisions_written,
        })


def iter_trace_decisions(paths, include_forced=False):
    """Yield portable records from multiple JSONL files in lexical order."""
    for path in sorted(map(Path, paths)):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("trace_schema") != TRACE_SCHEMA_VERSION:
                    raise ValueError("Unsupported trace schema")
                if record.get("type") == "decision" and (
                    include_forced or len(record["legal_cards"]) > 1
                ):
                    yield record
