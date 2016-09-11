from WhistPlayer import *

class RealHandRandomPlayer(WhistPlayer):
	def __init__(self, playerNumber, specificcards = [], strategy= "random"):
        self.strategy = strategy
        self.cardsLeftInHand = 7
        self.possiblehand = [ str(x) + 'h' for x in range( 2, 15 )]
        self.possiblehand += [ str(x) + 'd' for x in range( 2, 15 )]
        self.possiblehand += [ str(x) + 'c' for x in range( 2, 15 )]
        self.possiblehand += [ str(x) + 's' for x in range( 2, 15 )]
            
        self.realHand = specificcards

        self.playerNumber = playerNumber
        self.canBidZero = True
        self.zeroCounter = 0
        self.monteCarloNumber = 600

    def getRealPossibleMoves(self,pile):
        if len(pile) > 0:
            ledCards = cards_of_suit(self.realHand, pile[0][-1])
            if len(ledCards) > 0:
                return ledCards
            else:
                return self.realHand
        else:
            return self.realHand

    def make_real_move(self, pile, trumpsuit, fullGameObject = None):
        possibleMoves = self.getPossibleMoves(pile)
        realMoves = self.getRealPossibleMoves(pile)
        print " ------------ Player: ", self.playerNumber, " -----------"
        print "Trumps: ",
        print trumpsuit
        print "All Possible Moves: ",
        print possibleMoves
       	print "All Real Possible Moves: ",
        print realMoves
        print "Cards in pile: ",
        print pile
        if type(realMoves) is str:
            card = realMoves
        else:
            card = random.choice(realMoves)
        if self.play_card(card):
            self.cardSuitCheck(pile, card)
        return card
