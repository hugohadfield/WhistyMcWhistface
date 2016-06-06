
import copy
from WhistLib import *
import TrumpPicker
import sys
import traceback

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

    def remove_suit(self,card):
        suit = card[-1]
        cardsToRemove = cards_of_suit(self.possiblehand, suit)
        self.remove_possible(cardsToRemove)
    
    def make_move(self, pile, trumpsuit, fullGameObject = None):

        if self.strategy == "randomControlled":
            card = self.makeRandomControlledMove(pile,trumpsuit)
        elif self.strategy == "randomUncontrolled":
            card = self.makeRandomUncontrolledMove(pile,trumpsuit)
        elif self.strategy == "basic":
            card = self.makeBasicMove(pile,trumpsuit)
        elif self.strategy == "manualUncontrolled":
            card = self.makeManualUnControlledMove(pile,trumpsuit)
        elif self.strategy =="advanced":
            card = self.makeAdvancedMove(pile,trumpsuit,fullGameObject)
        return card

    def getPossibleMoves(self,pile):
        if len(pile) > 0:
            ledCards = cards_of_suit(self.possiblehand, pile[0][-1])
            if len(ledCards) > 0:
                return ledCards
            else:
                return self.possiblehand
        else:
            return self.possiblehand

    def makeRandomControlledMove(self,pile,trumpsuit):
        possibleMoves = self.getPossibleMoves(pile)
        if type(possibleMoves) is str:
            card = possibleMoves
        else:
            card = random.choice(possibleMoves)
        if self.play_card(card):
            return card

    def makeRandomUncontrolledMove(self,pile,trumpsuit):
        if type(self.possiblehand) is str:
            card = self.possiblehand
        else:
            card = random.choice(self.possiblehand)
        if self.play_card(card):
            return card

    def makeManualControlledMove(self,pile,trumpsuit):
        print " "
        print "Manual card choice required"
        while True:
            print "Trumps: ",
            print trumpsuit
            print "Legal Moves: ",
            print self.getPossibleMoves(pile)
            print "Cards in pile: ",
            print pile
            cardToPlay = raw_input('Enter your card choice: ').rstrip()
            if cardToPlay in self.getPossibleMoves(pile):
                if self.play_card(cardToPlay):
                    return cardToPlay
                else:
                    print "Invalid card choice, try again"
            else:
                print "Invalid card choice, try again"

    def makeManualUnControlledMove(self,pile,trumpsuit):
        print " "
        print "Manual card choice required"
        while True:
            print "Trumps: ",
            print trumpsuit
            print "All Possible Moves: ",
            print self.possiblehand
            print "Cards in pile: ",
            print pile
            cardToPlay = raw_input('Enter your card choice: ').rstrip()
            if self.play_card(cardToPlay):
                return cardToPlay
            else:
                print "Invalid card choice, try again"

    def makeBasicMove(self,pile,trumpsuit):
        return makeRandomMove(pile,trumpsuit)

    def makeAdvancedMove(self,pile,trumpsuit,fullGameObject):

        currentPossibleMoves = self.getPossibleMoves(pile)

        print "Trumps: ",
        print trumpsuit
        print "All Cards In Hand: ",
        print self.possiblehand
        print "Legal Moves: ",
        print currentPossibleMoves
        print "Cards in pile: ",
        print pile
        if currentPossibleMoves is not str:
            if len(currentPossibleMoves) > 1:
                # Project n steps into the future and evaluate the quality of the move
                monteCarloNumber = 1500
                thisPlayerCardRanking = []
                
                for specificCard in currentPossibleMoves:
                    # We suppress the monte carlo chatter
                    sys.stdout = NullIO()
                    [outputScores, errorCounter] = self.cardPlayMonteCarlo(specificCard,monteCarloNumber,fullGameObject)
                    sys.stdout = sys.__stdout__
                    thisPlayerCardRanking.append( outputScores[self.playerNumber] )
                    print errorCounter,
                print " "
                # Get the maximum value and expected point ranking
                print thisPlayerCardRanking
                [expectedPointGain, cardIndex] = max_and_index(thisPlayerCardRanking)

                # Play the card
                cardToPlay = currentPossibleMoves[cardIndex]
            else:
                cardToPlay = currentPossibleMoves[0]
        else:
            cardToPlay = currentPossibleMoves
        if self.play_card(cardToPlay):
            return cardToPlay

    def cardPlayMonteCarlo(self,specificCard,monteCarloNumber,fullGameObject):

        averageScores = []
        for i in range(0,fullGameObject.numberofplayers):
            averageScores.append( 0.0 )

        # Keep track of any game errors and use them to correct score estimate
        errorCounter = 0
        for monteIterator in range(0,monteCarloNumber): 
            # Make a copy of the current game object
            tempGame = copy.deepcopy( fullGameObject )

            # Merge the score output 
            for i in range(0,tempGame.numberofplayers):
                tempGame.scores[i] = 0.0

            # Revert all players to random
            for p in tempGame.players:
                if p.strategy == "advanced":
                    p.strategy = "randomControlled"
                else:
                    p.strategy = "randomUncontrolled"

            try:
                # Play the whole round
                tempGame.playPartialTrick(specificCard)
                tempGame.playPartialRound()
                
                # Merge the score output 
                for i in range(0,tempGame.numberofplayers):
                    averageScores[i] = averageScores[i] + float(tempGame.scores[i])
            except:
                errorCounter = errorCounter + 1

        # Convert to proper average
        for i in range(0,tempGame.numberofplayers):
            averageScores[i] = (1/float(monteCarloNumber - errorCounter))*averageScores[i]
        return averageScores, errorCounter

    def pick_trumps(self):
        return TrumpPicker.picktrump(self.possiblehand)

    def make_bid(self, nplayers, ncards, trumpsuit, previousbids):

        if self.strategy == "randomControlled":
            thisbid = self.makeAverageBid(nplayers,ncards)
        elif self.strategy == "randomUncontrolled":
            thisbid = self.makeAverageBid(nplayers,ncards)
        elif self.strategy == "basic":
            thisbid = self.makeAverageBid(nplayers,ncards)
        elif self.strategy == "manualUncontrolled":
            thisbid = self.makeManualBid()
        elif self.strategy =="advanced":
            thisbid = self.makeAdvancedBid( nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        return thisbid
    
    def makeAverageBid(self,nplayers,ncards):
        thisbid = int(round(ncards/float(nplayers)))
        return thisbid

    def makeManualBid(self):
        thisbid = int(raw_input())
        return thisbid
    
    def makeAdvancedBid(self, nplayers, ncards, trumpsuit, previousbids):
        cardsInHand = self.possiblehand
        vicProbList = self.computeProbability1CardVictory(nplayers, cardsInHand, ncards, trumpsuit)
        bidPDF = monte_carlo_pdfify(vicProbList,5000)
        [confidence, thisbid] = max_and_index(bidPDF)
        print "Bid confidence: ", confidence
        return thisbid

    def computeProbability1CardVictory(self, nplayers, cardsInHand, ncards, trumpsuit):
        # I know this is wrong, but not very wrong, needs some sampling without replacement stuff...
        outputProbs = []
        for card in cardsInHand:
            probabilityOfWin1v1 = 1 - ( len(  card_beating_list(card, trumpsuit)  ) / float(52-1) )
            probabilityTotalVictory = probabilityOfWin1v1**nplayers
            outputProbs.append( probabilityTotalVictory )
        return outputProbs