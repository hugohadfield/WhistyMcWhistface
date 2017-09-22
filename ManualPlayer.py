from WhistPlayer import *
from RandomPlayer import *
GRAPHICAL_INPUT = False
try:
    import easygui
    GRAPHICAL_INPUT = True
except:
    print('easygui not available, defaulting to text input')

class ManualPlayer(Player):

    def get_manual_input(self,message,possible_choices):
        cardToPlay = ''
        if GRAPHICAL_INPUT:
            cardToPlay = easygui.choicebox(msg=message, 
                title='Player '+str(self.playerNumber), 
                choices = possible_choices).rstrip()
        else:
            cardToPlay = raw_input('Enter your card choice: ').rstrip()
        return cardToPlay

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
            message = 'Player ' + str(self.playerNumber) +'. Enter your card choice: '
            cardToPlay = self.get_manual_input(message, self.possibleHand)
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

    def pick_trumps(self, nplayers):
        print "Enter trump: "
        trump = raw_input().rstrip()
        return trump

    def generateModelPlayer(self):
        modelPlayer = RandomPlayer(self.playerNumber, self.possibleHand)
        modelPlayer.canBidZero = self.canBidZero
        modelPlayer.zeroCounter = self.zeroCounter
        modelPlayer.cardsLeftInHand = self.cardsLeftInHand
        modelPlayer.realHand = [i for i in self.realHand]
        return copy.deepcopy(modelPlayer)