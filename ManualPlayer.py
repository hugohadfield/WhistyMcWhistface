from WhistPlayer import *

class ManualPlayer(Player):

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        print " "
        print "Manual card choice required"
        while True:
            print "Trumps: ",
            print trumpsuit
            print "All Possible Moves: ",
            print self.possibleHand
            print "Cards in pile: ",
            print pile
            cardToPlay = raw_input('Enter your card choice: ').rstrip()
            if self.playCard(cardToPlay):
                self.cardSuitCheck(pile, cardToPlay)
                return cardToPlay
            else:
                print "Invalid card choice, try again"

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        print "Enter bid: "
        thisbid = int(raw_input())
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def pick_trumps(self):
        print "Enter trump: "
        trump = raw_input().rstrip()
        return trump
