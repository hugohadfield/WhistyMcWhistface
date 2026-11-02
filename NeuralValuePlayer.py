"""Whist player using a trained information-safe distributional MLP.

Inherits *the identical* MonteCarloPlayer bid and trump methods for fair
head-to-head card-play comparison. The neural network is ONLY used to choose
a legal card, never to choose a bid/trump or to inspect hidden opponents.

To use in a Game config:
 {"class":"NeuralValuePlayer","playerNumber":0,
  "additionalParameters":{"checkpoint":"runs/whist-v1/best.pt","device":"cpu"}}
"""
from pathlib import Path

from MonteCarloPlayer import MonteCarloPlayer


class NeuralValuePlayer(MonteCarloPlayer):
    def __init__(self, playerNumber, specificCards=None,
                 additionalParameters=None, checkpoint=None, device="cpu"):
        # MC bid/trump implementation is reused; don't treat checkpoint as
        # Monte Carlo-specific parameters.
        super().__init__(playerNumber, specificCards=specificCards)
        params = dict(additionalParameters or {})
        self.checkpoint = str(checkpoint or params.get("checkpoint", ""))
        self.device = device if checkpoint is not None else params.get("device", device)
        if not self.checkpoint:
            raise ValueError("NeuralValuePlayer needs a trained checkpoint")
        self._model = None

    def _get_model(self):
        if self._model is None:
            from alphawhist.learning.checkpoint import load_checkpoint
            model, _ = load_checkpoint(self.checkpoint, device=self.device)
            self._model = model
        return self._model

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        moves = self.getRealPossibleMoves(pile)
        if not moves:
            raise ValueError("No legal cards")
        if len(moves) == 1:
            card = moves[0]
        else:
            if fullGameObject is None:
                raise ValueError("Neural decision requires the player's Game observation")
            from alphawhist.learning.encoding import CARDS, encode_decision
            observation = fullGameObject.observe(self.playerNumber)
            encoded = encode_decision(observation)
            pred = self._get_model().predict_encoded(encoded)
            card = CARDS[int(pred.best_card_ids()[0].item())]
            if card not in moves:
                raise AssertionError("Neural policy returned an illegal card")
        if not self.playCard(card):
            raise ValueError("Selected neural card disappeared from hand")
        self.cardSuitCheck(pile, card)
        return card

    def export_policy_state(self):
        # Do not include filesystem path or model weights in observation
        # policy memory; it has no bearing on optimal Whist decisions.
        result = super().export_policy_state()
        result["neural_value_policy"] = True
        return result
