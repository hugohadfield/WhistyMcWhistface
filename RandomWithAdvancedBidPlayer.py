from alphawhist.core.WhistPlayer import *
import copy

class RandomWithAdvancedBidPlayer(Player):

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
            card = random.choice(possibleMoves)
        if self.playCard(card):
            self.cardSuitCheck(pile, card)
            return card
        else:
            print("Error playing card")

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        thisbid = self.makeAdvancedBid(nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    
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
        return copy.deepcopy(self)