from RandomPlayer import *
from alphawhist.core.FastRollout import DEFAULT_ROLLOUTS
import random
import sys
import traceback
from operator import add

class MonteCarloPlayer(Player):
    # Defaults also apply when no additionalParameters are supplied.
    monteCarloNumber = DEFAULT_ROLLOUTS
    monteCarloWorkers = 1
    monteCarloPolicy = "random"


    def handleAdditionalParamaters(self, parameters):
        self.monteCarloNumber = int(parameters.get("monteCarloNumber", DEFAULT_ROLLOUTS))
        self.monteCarloWorkers = int(parameters.get("monteCarloWorkers", 1))
        self.monteCarloPolicy = parameters.get("monteCarloPolicy", "random")
        if self.monteCarloNumber < 1 or self.monteCarloWorkers < 1:
            raise ValueError("Monte Carlo rollouts and workers must be positive")
        if self.monteCarloPolicy not in ("random", "highest", "bid_aware_2"):
            raise ValueError("Unknown Monte Carlo rollout policy")
        return True

    def pick_trumps(self, nplayers):
        # Dealer chooses trumps before any bids have been made.
        return self.alternateTrumpPicker(nplayers)

    def alternateTrumpPicker(self, nplayers):
        """Active exact-DP trump selector (replaces the dormant legacy version)."""
        return best_trump_by_expected_score(
            self.real_hand, nplayers, can_bid_zero=self.can_bid_zero
        )

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        """Evaluate every legal move with shared hidden-world rollouts."""
        from alphawhist.core.FastRollout import evaluate_actions

        moves = self.getRealPossibleMoves(pile)
        if not moves:
            raise ValueError("No legal moves")
        if len(moves) == 1:
            card = moves[0]
        else:
            if fullGameObject is None:
                raise ValueError("Monte Carlo move selection requires a Game")
            observation = fullGameObject.observe(self.playerNumber)
            result = evaluate_actions(
                observation, moves,
                n_rollouts=self.monteCarloNumber,
                seed=random.getrandbits(64),
                workers=self.monteCarloWorkers,
                objective="score",
                policy=self.monteCarloPolicy,
            )
            card = result.best_action()
        if not self.playCard(card):
            raise ValueError("Selected card is no longer in hand")
        self.cardSuitCheck(pile, card)
        return card

    def cardPlayMonteCarlo(self, specificCard, monteCarloNumber, fullGameObject):
        """Compatibility wrapper, now using the history-free simulator."""
        from alphawhist.core.FastRollout import evaluate_actions

        if fullGameObject is None:
            raise ValueError("Monte Carlo evaluation requires a Game")
        result = evaluate_actions(
            fullGameObject.observe(self.playerNumber),
            [specificCard],
            n_rollouts=monteCarloNumber,
            seed=random.getrandbits(64),
            workers=self.monteCarloWorkers,
            objective="score",
            policy=self.monteCarloPolicy,
        )
        return list(result.mean_round_scores[0]), 0

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        #thisbid = self.makeAverageBid( nplayers, ncards, trumpsuit, previousbids)
        thisbid = self.makeAdvancedBid(nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def makeAverageBid(self, nplayers, ncards, trumpsuit, previousbids):
        originalbid = int(round(ncards / float(nplayers)))
        validBids = self.getValidBidOptions(nplayers, ncards, previousbids)
        for i in range(0, ncards+1):
            thisbid = (originalbid + i) % (ncards+1)
            if thisbid in validBids:
                return thisbid
        print("Bid Error")
        print(validBids)
        print(originalbid)
        print(ncards)

    def makeAdvancedBid(self, nplayers, ncards, trumpsuit, previousbids):

        bidPosition = len(previousbids)

        cardsInHand = self.real_hand
        leadingProbList, followingProbList = estimate_card_win_probabilities(
            cardsInHand, nplayers, ncards, trumpsuit
        )
        bidPDF = exact_bid_trick_pdf(leadingProbList, followingProbList, bidPosition)

        validBids = self.getValidBidOptions(nplayers, ncards, previousbids)
        for i in range(0, ncards + 1):
            if i not in validBids:
                bidPDF[i] = -1.0

        [confidence, thisbid] = max_and_index(bidPDF)
        print("Bid: ", thisbid)
        print("Bid confidence: ", confidence)
        return thisbid

    def computeLeadingCardVictory(self, nplayers, cardsInHand, ncards, trumpsuit):
        return estimate_card_win_probabilities(cardsInHand, nplayers, ncards, trumpsuit)[0]

    def computeFollowingCardVictory(self, nplayers, cardsInHand, ncards, trumpsuit):
        return estimate_card_win_probabilities(cardsInHand, nplayers, ncards, trumpsuit)[1]

    def generateModelPlayer(self):
        modelPlayer = RandomPlayer(self.playerNumber, self.possible_hand)
        modelPlayer.resetZeroCounter(self.zero_bid_count)
        modelPlayer.monteCarloNumber = self.monteCarloNumber
        modelPlayer.set_hands(self.real_hand, self.possible_hand)
        return copy.deepcopy(modelPlayer)