
from WhistLib import *
import TrumpPicker

class Player():
    def __init__(self, playerNumber, specificcards = [], strategy= "random"):
        self.strategy = strategy
        if not specificcards:
            self.possiblehand = [ str(x) + 'h' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 'd' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 'c' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 's' for x in range( 2, 15 )]
        else:
            self.possiblehand = specificcards
        self.playerNumber = playerNumber

    def remove_possible(self, cards):
        if type(cards) is str:
            cleanCard = cards.rstrip()
            if cleanCard in self.possiblehand:
                self.possiblehand.remove(cleanCard)
        else:
            for card in cards:
                cleanCard = card.rstrip()
                if cleanCard in self.possiblehand:
                    self.possiblehand.remove(cleanCard)
                
    def play_card(self, card):
        cleanCard = card.rstrip()
        print ["Removing: ", cleanCard]
        if cleanCard in self.possiblehand:
            self.remove_possible(cleanCard)
            return 1
        else:
            return 0
    
    def make_move(self, pile, trumpsuit, fullGameObject = None):
        if self.strategy == "random":
            card = self.makeRandomMove(pile,trumpsuit)
        elif self.strategy == "basic":
            card = self.makeBasicMove(pile,trumpsuit)
        elif self.strategy == "manual":
            card = self.makeManualMove(pile,trumpsuit)
        elif self.strategy =="advanced":
            card = self.makeAdvancedMove(pile,trumpsuit,fullGameObject)
        return card

    def makeRandomMove(self,pile,trumpsuit):
        if len(pile) > 0:
            thisuitcards = cards_of_suit(self.possiblehand , pile[0][-1])
            if len(thisuitcards) > 0:
                card = random.choice(thisuitcards)
            else:
                card = random.choice(self.possiblehand)
        else:
            card = random.choice(self.possiblehand)
        if self.play_card(card):
            return card

    def makeManualMove(self,pile,trumpsuit):
        print "Manual card choice required"
        while True:
            print "Trumps: ",
            print trumpsuit
            print "Cards in hand: ",
            print self.possiblehand
            print "Cards in pile: ",
            print pile
            cardToPlay = raw_input('Enter your card choice: ')
            if self.play_card(cardToPlay.rstrip()):
                return cardToPlay
            else:
                print "Invalid card choice, try again"

    def makeBasicMove(self,pile,trumpsuit):
        return makeRandomMove(pile,trumpsuit)

    def makeAdvancedMove(self,pile,trumpsuit,fullGameObject):
        # Project n steps into the future and evaluate the quality of the move
        for specificCard in self.possiblehand:
            # Make a copy of the current game object
            tempGame = copy.deepcopy( fullGameObject )
            tempGame.playPartialTrick(specificCard)
            tempGame.playPartialRound()
            tempGame.scores
        # Compare how many times you win for each of the possible moves
        return makeRandomMove(pile)
    
    def make_bid(self, trumpsuit):
        thisbid = 1
        self.bid = thisbid
        return thisbid
        
    def pick_trumps(self):
        return TrumpPicker.picktrump(self.possiblehand)


class BotModel():
    def __init__(self, thisplayercards):

        # Game related variables
        self.numberofplayers = 3
        self.round = 0
        self.cardsplayed = 0
        self.dealer = 0
        self.trumps = 'h'
        self.players = []

        # The player we play as
        thisplayer = Player(thisplayercards)
        self.players.append(copy.deepcopy(thisplayer))
        
        # The other players
        for i in range(0,self.numberofplayers-1):
            thisplayer = Player()
            thisplayer.remove_possible(thisplayercards)
            self.players.append(copy.deepcopy(thisplayer))

