"""Stand-alone bid_aware_2 card player, matching the rollout heuristic exactly.

Everything except the choice between two legal cards from a two-card hand
comes from RandomPlayer: identical bidding, trump choice and random plays.
No Monte Carlo is performed. The endgame heuristic can see only the actor's
own cards, public bids/trick counts/current pile, and trumps.
"""
from types import SimpleNamespace
import random

from RandomPlayer import RandomPlayer
from alphawhist.core.EndgamePolicy import pick_bid_aware_two_card
from alphawhist.core.FastRollout import CARD_ID, CARDS, SUITS, hand_mask


class BidAware2Player(RandomPlayer):
    """RandomPlayer with a bid-aware (softmax) two-card endgame."""

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        legal_cards = self.getRealPossibleMoves(pile)
        # Exactly the RandomPlayer implementation outside two-card choices.
        if len(self.real_hand) != 2 or len(legal_cards) != 2:
            return super().makeMove(pile, trumpsuit, fullGameObject)

        if fullGameObject is None:
            raise ValueError("BidAware2Player needs public game observation")

        # InformationState is *actor-specific*, never contains other hands.
        observation = fullGameObject.observe(self.playerNumber)
        if (observation.observer_id != self.playerNumber
                or observation.turn != self.playerNumber
                or observation.trump != trumpsuit
                or tuple(card for _, card in observation.pile) != tuple(pile)
                or set(observation.own_hand) != set(self.real_hand)):
            raise AssertionError("Public observation disagrees with acting player")

        state = SimpleNamespace(
            # EndgamePolicy only inspects our own hand and player count.
            hands=tuple(
                hand_mask(observation.own_hand) if i == self.playerNumber else 0
                for i in range(observation.num_players)
            ),
            turn=self.playerNumber,
            pile=[CARD_ID[card] for card in pile],
            bids=tuple(observation.bids),
            tricks_won=tuple(observation.tricks_won),
            trump=SUITS.index(trumpsuit),
        )
        chosen_id = pick_bid_aware_two_card(state, hand_mask(legal_cards), random)
        card = CARDS[chosen_id]
        if card not in legal_cards:
            raise AssertionError("Bid-aware heuristic produced an illegal play")
        if not self.playCard(card):
            raise AssertionError("Selected card was missing from player's own hand")
        self.cardSuitCheck(pile, card)
        return card
