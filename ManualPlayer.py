from RandomPlayer import *
GRAPHICAL_INPUT = False
try:
    import easygui
    GRAPHICAL_INPUT = True
except:
    print('easygui not available, defaulting to text input')

class ManualPlayer(Player):

    def get_manual_input(self,message,possible_choices):
        translated_choices = [translate_card_name(i) for i in possible_choices]
        cardToPlay = ''
        if GRAPHICAL_INPUT:
            cardToPlay = easygui.choicebox(msg=message, 
                title='Player '+str(self.playerNumber), 
                choices = translated_choices).rstrip()
        else:
            cardToPlay = input('Enter your card choice: ').rstrip()
        translated_card = translate_inv_card_name(cardToPlay)
        return cardToPlay

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        print(" ")
        print("Manual card choice required")
        while True:
            print("Trumps: ", end=' ')
            print(trumpsuit)
            print("All Possible Moves: ", end=' ')
            print(self.possible_hand)
            print("Cards in pile: ", end=' ')
            print(pile)
            message = 'Player ' + str(self.playerNumber) +'. Enter your card choice: '
            cardToPlay = self.get_manual_input(message, self.possible_hand)
            if self.playCard(cardToPlay):
                self.cardSuitCheck(pile, cardToPlay)
                return cardToPlay
            else:
                print("Invalid card choice, try again")

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        thisbid = None
        if GRAPHICAL_INPUT:
            thisbid = easygui.integerbox(msg='Choose bid', 
                title='Choose bid', 
                lowerbound=0, upperbound=10)
        else:
            print("Enter bid: ")
            thisbid = int(input())
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def pick_trumps(self, nplayers):
        trump = ''
        possible_trumps = ['h','s','d','c']
        while trump not in possible_trumps:
            if GRAPHICAL_INPUT:
                trump = easygui.choicebox(msg='Choose trump', 
                    title='Choose trump', 
                    choices = possible_trumps).rstrip()
            else:
                print("Enter trump: ")
                trump = input().rstrip()
        return trump

    def generateModelPlayer(self):
        modelPlayer = RandomPlayer(self.playerNumber, self.possible_hand)
        modelPlayer.resetZeroCounter(self.zero_bid_count)
        modelPlayer.real_hand = [i for i in self.real_hand]
        return copy.deepcopy(modelPlayer)