"""Optional bid-aware 3-card player using analytical suit + joint-trick model.

Preserves exact RandomPlayer bidding/trump selection and baseline
BidAware3Player decisions for two cards and all earlier hands.
Only three-card decisions with >=2 legal moves are different.
"""
import random

from BidAware3Player import BidAware3Player
from alphawhist.core.AnalyticEndgame import select_analytical_three_card


class AnalyticBidAware3Player(BidAware3Player):
    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        legal = self.getRealPossibleMoves(pile)
        if len(self.real_hand) != 3 or len(legal) < 2:
            return super().makeMove(pile,trumpsuit,fullGameObject)
        if fullGameObject is None:
            raise ValueError("Three-card analytic policy needs an observation")
        observation=fullGameObject.observe(self.playerNumber)
        if (observation.observer_id!=self.playerNumber
                or observation.turn!=self.playerNumber
                or observation.trump!=trumpsuit
                or tuple(c for _,c in observation.pile)!=tuple(pile)
                or set(observation.own_hand)!=set(self.real_hand)):
            raise AssertionError("Mismatch in observer-private public game state")
        self.three_card_choices=getattr(self,"three_card_choices",0)+1
        card=select_analytical_three_card(observation,legal,random)
        if card not in legal or not self.playCard(card):
            raise AssertionError("Analytical policy selected an illegal card")
        self.cardSuitCheck(pile,card)
        return card
