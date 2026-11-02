"""A playable Monte Carlo bot whose own future decisions use a frozen neural net.

Both bidding and trump selection are inherited unchanged from MonteCarloPlayer.
At a nonforced card decision, evaluate each legal card over sampled hidden
worlds, then use the frozen neural policy for our own subsequent simulated
turns. Opponents use the selected continuation policy: bid_aware_2 (default) or
the SAME neural checkpoint from each opponent's own sampled perspective.
The neural policy never observes any other simulated player's private cards.
"""
import random

from MonteCarloPlayer import MonteCarloPlayer


class NeuralRolloutMonteCarloPlayer(MonteCarloPlayer):
    def __init__(self, playerNumber, specificCards=None,
                 additionalParameters=None, checkpoint=None,
                 rollouts=None, opponent_policy=None, worlds_per_batch=16):
        super().__init__(playerNumber, specificCards=specificCards)
        params = dict(additionalParameters or {})
        self.checkpoint = str(checkpoint or params.get("checkpoint", ""))
        self.rollouts = int(rollouts if rollouts is not None else
                            params.get("rollouts", 128))
        self.opponent_policy = (opponent_policy or
                                params.get("opponent_policy", "bid_aware_2"))
        self.worlds_per_batch = int(params.get(
            "worlds_per_batch", worlds_per_batch))
        if not self.checkpoint:
            raise ValueError("NeuralRolloutMonteCarloPlayer needs a checkpoint")
        if self.rollouts < 1 or self.worlds_per_batch < 1:
            raise ValueError("Rollout budget and batch size must be positive")
        if self.opponent_policy not in ("random", "highest", "bid_aware_2", "neural"):
            raise ValueError("Unknown opponent policy")

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        legal = self.getRealPossibleMoves(pile)
        if not legal:
            raise ValueError("No legal card")
        if len(legal) == 1:
            card = legal[0]
        else:
            if fullGameObject is None:
                raise ValueError("Neural MC needs the actor's Game observation")
            from alphawhist.learning.neural_rollout import evaluate_neural_self
            observation = fullGameObject.observe(self.playerNumber)
            evaluated = evaluate_neural_self(
                observation, legal, checkpoint=self.checkpoint,
                n_rollouts=self.rollouts, seed=random.getrandbits(64),
                workers=1, opponent_policy=self.opponent_policy,
                worlds_per_batch=self.worlds_per_batch,
            )
            card = evaluated.best_action()
            if card not in legal:
                raise AssertionError("Neural MC returned an illegal card")
        if not self.playCard(card):
            raise ValueError("Selected card disappeared from hand")
        self.cardSuitCheck(pile, card)
        return card
