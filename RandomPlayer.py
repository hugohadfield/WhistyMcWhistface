from WhistPlayer import *

class RandomPlayer(Player):
    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        possibleMoves = self.getPossibleMoves(pile)
        print " ------------ Player: ", self.playerNumber, " -----------"
        print "Trumps: ",
        print trumpsuit
        print "All Possible Moves: ",
        print possibleMoves
        print "Cards in pile: ",
        print pile
        if type(possibleMoves) is str:
            card = possibleMoves
        else:
            card = random.choice(possibleMoves)
        if self.playCard(card):
            self.cardSuitCheck(pile, card)
            return card

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        thisbid = self.makeAverageBid( nplayers, ncards, trumpsuit, previousbids)
        return thisbid

    def makeAverageBid(self, nplayers, ncards, trumpsuit, previousbids):
        originalbid = int(round(ncards / float(nplayers)))
        validBids = self.getValidBidOptions(nplayers, ncards, previousbids)
        for i in range(0, ncards):
            thisbid = (originalbid + i) % ncards
            if thisbid in validBids:
                return thisbid