from WhistPlayer import *
import copy

class RandomPlayer(Player):

    def pick_trumps(self, numberOfPlayers):
        thistrump = TrumpPicker.picktrump(self.possibleHand)
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
        thisbid = self.makeAverageBid( nplayers, ncards, trumpsuit, previousbids)
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

    def generateModelPlayer(self):
        return copy.deepcopy(self)