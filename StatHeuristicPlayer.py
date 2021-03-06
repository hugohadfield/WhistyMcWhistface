from alphawhist.core.WhistPlayer import *
import copy


class StatHeuristicPlayer(Player):
    def pick_trumps(self, nplayers):
        thistrump = TrumpPicker.picktrump(self.real_hand)
        return thistrump

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        possibleMoves = self.getRealPossibleMoves(pile)
        print(" ------------ Player: ", self.playerNumber, " -----------")
        print("Trumps: ", end=' ')
        print(trumpsuit)
        print("All Real Possible Moves: ", end=' ')
        print(possibleMoves)
        print("Cards in pile: ", end=' ')
        print(pile)
        if type(possibleMoves) is str:
            card = possibleMoves
        else:
            card = self.makeRewardBasedMove( possibleMoves )
        if self.playCard(card):
            self.cardSuitCheck(pile, card)
            return card
        else:
            print("Error playing card")

    def makeRewardBasedMove(self, possibleMoves):
        return card

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        thisbid = self.makeAdvancedBid(nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def makeAdvancedBid(self, nplayers, ncards, trumpsuit, previousbids):

        bidPosition = len(previousbids)

        cardsInHand = self.real_hand
        leadingProbList = self.computeLeadingCardVictory(nplayers, cardsInHand, ncards, trumpsuit)
        followingProbList = self.computeFollowingCardVictory(nplayers, cardsInHand, ncards, trumpsuit)
        bidPDF = mini_monte_sim(leadingProbList, followingProbList, bidPosition, 20000)

        validBids = self.getValidBidOptions(nplayers, ncards, previousbids)
        for i in range(0, ncards):
            if i not in validBids:
                bidPDF[i] = -1.0

        [confidence, thisbid] = max_and_index(bidPDF)
        print("Bid: ", thisbid)
        print("Bid confidence: ", confidence)
        return thisbid

    def computeExpectedRewardForLedCard(self, leadCard, nplayers, cardsInHand, ncards, trumpsuit, bid, pile):
        outputProbs = []
        handSize = len(cardsInHand)
        for card in cardsInHand:
            # All cards that beat this card, assuming this card leads
            cardlist = card_beating_list(card, trumpsuit)
            # All cards that beat this card and are not in my hand
            possibleOppList = list(set(cardsInHand) ^ set(cardlist))
            # Probability that an opposition player does not have one of these cards
            probabilityOfWin1v1 = 1 - (len(possibleOppList) / float(52 - handSize))
            # I know this is wrong, but not very wrong, needs some sampling without replacement stuff...
            # Probability that no opposition players have one of these cards
            probabilityTotalVictory = probabilityOfWin1v1 ** (nplayers - 1)
            outputProbs.append(probabilityTotalVictory)
        return outputProbs

    def computeExpectedRewardForFollowedCard(self, nplayers, cardsInHand, ncards, trumpsuit):
        outputProbs = []
        handSize = len(cardsInHand)
        for card in cardsInHand:
            # All cards that beat this card, assuming this card leads
            cardlist = card_beating_list(card, trumpsuit)
            # All cards that beat this card and are not in my hand
            possibleOppList = list(set(cardsInHand) ^ set(cardlist))
            # Probability that an opposition player does not have one of these cards
            probabilityOfWin1v1 = 1 - (len(possibleOppList) / float(52 - handSize))
            # I know this is wrong, but not very wrong, needs some sampling without replacement stuff...
            # Probability that no opposition players have one of these cards
            probabilityTotalVictory = probabilityOfWin1v1 ** (nplayers - 1)
            outputProbs.append(probabilityTotalVictory)
        return outputProbs

    def generateModelPlayer(self):
        return copy.deepcopy(self)