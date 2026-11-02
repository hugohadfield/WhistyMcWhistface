"""Cheap bid-aware three-card player: random until three cards remain.

At three cards, score each legal first card by its chance of winning this
trick and the softmax-weighted expected value of the remaining two-card
policy. Thereafter, retain BidAware2Player's existing exact heuristic.
Bidding and trump selection are inherited unchanged from RandomPlayer.
"""
from types import SimpleNamespace
import random

from BidAware2Player import BidAware2Player
from alphawhist.core.EndgamePolicy import pick_bid_aware_three_card
from alphawhist.core.FastRollout import CARDS, CARD_ID, SUITS, hand_mask


class BidAware3Player(BidAware2Player):
    """Three-card lookahead followed by unchanged bid-aware-2 endgame."""

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        legal_cards = self.getRealPossibleMoves(pile)
        if len(self.real_hand) != 3 or len(legal_cards) < 2:
            return super().makeMove(pile, trumpsuit, fullGameObject)
        if fullGameObject is None:
            raise ValueError("BidAware3Player needs public game observation")
        observation = fullGameObject.observe(self.playerNumber)
        if (observation.observer_id != self.playerNumber
                or observation.turn != self.playerNumber
                or observation.trump != trumpsuit
                or tuple(card for _, card in observation.pile) != tuple(pile)
                or set(observation.own_hand) != set(self.real_hand)):
            raise AssertionError("Public observation disagrees with acting player")

        # Only the acting player's hand is visible. The evaluation NEVER
        # receives the Game or its opponents' real cards.
        state = SimpleNamespace(
            hands=tuple(
                hand_mask(observation.own_hand) if i == self.playerNumber else 0
                for i in range(observation.num_players)),
            turn=self.playerNumber,
            pile=[CARD_ID[card] for card in pile],
            bids=tuple(observation.bids),
            tricks_won=tuple(observation.tricks_won),
            trump=SUITS.index(trumpsuit),
        )
        chosen_id = pick_bid_aware_three_card(
            state, hand_mask(legal_cards), random)
        card = CARDS[chosen_id]
        if card not in legal_cards or not self.playCard(card):
            raise AssertionError("Three-card heuristic produced an illegal card")
        self.cardSuitCheck(pile, card)
        return card
