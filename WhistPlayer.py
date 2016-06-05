
from WhistLib import *
import TrumpPicker

class Player():
    def __init__(self, specificcards = []):
        self.strategy = "random"
        if not specificcards:
            self.possiblehand = [ str(x) + 'h' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 'd' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 'c' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 's' for x in range( 2, 15 )]
        else:
            self.possiblehand = specificcards

    def remove_possible(self, cards):
        for card in cards:
            if card in self.possiblehand:
                self.possiblehand.remove(card)
                
    def play_card(self, card):
        if card in self.possiblehand:
            self.remove_possible(card)
            return 1
        else:
            return 0
    
    def make_move(self, pile):
        if self.strategy == "random":
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

        if self.strategy == "basic":
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

        if self.strategy == "manual":
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

