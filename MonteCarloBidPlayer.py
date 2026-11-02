"""Standard MonteCarloPlayer card play with MC-evaluated bidding.

Use additionalParameters:
  monteCarloNumber: "1024"        # real card-play search (unchanged)
  monteCarloPolicy: "bid_aware_2"
  bidRollouts: "1024"             # cheap complete-round bid simulations
"""
import random

from MonteCarloPlayer import MonteCarloPlayer
from alphawhist.core.BidRollout import evaluate_bids


class MonteCarloBidPlayer(MonteCarloPlayer):
    def __init__(self, playerNumber, specificCards=None, additionalParameters=None):
        # WhistPlayer.__init__ calls handleAdditionalParamaters before returning.
        self.bidRollouts = 1024
        super().__init__(playerNumber, specificCards=specificCards,
                         additionalParameters=additionalParameters)

    def handleAdditionalParamaters(self, parameters):
        super().handleAdditionalParamaters(parameters)
        self.bidRollouts = int(parameters.get("bidRollouts", 1024))
        if self.bidRollouts < 1:
            raise ValueError("bidRollouts must be positive")
        return True

    def makeBidWithObservation(self, nplayers, ncards, trumpsuit,
                               previousbids, observation):
        """Game passes a player-private, public-history-only observation."""
        if (observation.observer_id != self.playerNumber
                or observation.num_players != nplayers
                or observation.cards_per_player != ncards
                or observation.trump != trumpsuit
                or observation.own_hand != tuple(self.real_hand)):
            raise ValueError("Bidding observation mismatches player")
        allowed = self.getValidBidOptions(nplayers, ncards, previousbids)
        result = evaluate_bids(
            observation, allowed, n_rollouts=self.bidRollouts,
            seed=random.getrandbits(64),
        )
        bid = result.best_bid()
        self.bid = bid
        self.advanceZeroCounter(bid)
        return bid
