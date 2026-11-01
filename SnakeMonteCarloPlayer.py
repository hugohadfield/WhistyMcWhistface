from RandomPlayer import *
from alphawhist.core.FastRollout import DEFAULT_ROLLOUTS
import random
import sys
import traceback


class SnakeMonteCarloPlayer(Player):
    # Defaults also apply when no additionalParameters are supplied.
    monteCarloNumber = DEFAULT_ROLLOUTS
    monteCarloWorkers = 1


    def handleAdditionalParamaters(self, parameters):
        self.monteCarloNumber = int(parameters.get("monteCarloNumber", DEFAULT_ROLLOUTS))
        self.monteCarloWorkers = int(parameters.get("monteCarloWorkers", 1))
        if self.monteCarloNumber < 1 or self.monteCarloWorkers < 1:
            raise ValueError("Monte Carlo rollouts and workers must be positive")
        return True

    def pick_trumps(self, nplayers):
        # thistrump = self.alternateTrumpPicker(nplayers)
        thistrump = TrumpPicker.picktrump(self.real_hand)
        return thistrump

    def alternateTrumpPicker(self, nplayers):
        cardsInHand = self.real_hand
        ncards = len(self.real_hand)
        possibleTrumps = ['h', 'd', 'c', 's']
        confidenceList = []
        for trumpsuit in possibleTrumps:
            leadingProbList, followingProbList = estimate_card_win_probabilities(
                cardsInHand, nplayers, ncards, trumpsuit
            )
            bidPDF = mini_monte_sim(leadingProbList, followingProbList, 2, 50000)
            validBids = self.getValidBidOptions(nplayers, ncards, [])
            for i in range(0, ncards + 1):
                if i not in validBids:
                    bidPDF[i] = -1.0
            [confidence, thisbid] = max_and_index(bidPDF)
            confidenceList.append(confidence)
        print("Trump confidences: ", confidenceList)
        [maxConfidence, trumpIndex] = max_and_index(confidenceList)
        return possibleTrumps[trumpIndex]

    def snakeFunction(self, p_num, scores):
        not_my_scores = [x for i, x in enumerate(scores) if i != p_num]
        snake_val = scores[p_num] - max(not_my_scores)
        return snake_val

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
                objective="margin",
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
            objective="margin",
        )
        return list(result.mean_round_scores[0]), 0

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        # thisbid = self.makeAverageBid( nplayers, ncards, trumpsuit, previousbids)
        thisbid = self.makeAdvancedBid(nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def makeAverageBid(self, nplayers, ncards, trumpsuit, previousbids):
        originalbid = int(round(ncards / float(nplayers)))
        validBids = self.getValidBidOptions(nplayers, ncards, previousbids)
        for i in range(0, ncards + 1):
            thisbid = (originalbid + i) % (ncards + 1)
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
        bidPDF = mini_monte_sim(leadingProbList, followingProbList, bidPosition, 20000)

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
