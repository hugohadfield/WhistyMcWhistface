
import copy
from WhistLib import *
import TrumpPicker
import sys
import traceback
from operator import add

class Player():
    def __init__(self, playerNumber, specificcards = [], strategy= "random"):
        self.strategy = strategy
        self.cardsLeftInHand = 7
        if not specificcards:
            self.possiblehand = [ str(x) + 'h' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 'd' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 'c' for x in range( 2, 15 )]
            self.possiblehand += [ str(x) + 's' for x in range( 2, 15 )]
        else:
            self.possiblehand = specificcards
        self.playerNumber = playerNumber
        self.canBidZero = True
        self.zeroCounter = 0
        self.monteCarloNumber = 600


    def remove_possible(self, cards):
        newHand = [i for i in self.possiblehand]

        if type(cards) is str:
            cleanCard = cards.rstrip()
            if cleanCard in newHand:
                newHand.remove(cleanCard)
                self.setPossibleHand(newHand)
        else:
            for card in cards:
                cleanCard = card.rstrip()
                if cleanCard in newHand:
                    self.setPossibleHand(newHand)

                
    def play_card(self, card):
        cleanCard = card.rstrip()
        if cleanCard in self.possiblehand:
            print ["Removing: ", cleanCard]
            self.cardsLeftInHand = self.cardsLeftInHand - 1
            self.remove_possible(cleanCard)
            return True
        else:
            print " Card is not in hand "
            print cleanCard
            print self.possiblehand
            return False

    def remove_suit(self,card):
        suit = card[-1]
        cardsToRemove = cards_of_suit(self.possiblehand, suit)
        self.remove_possible(cardsToRemove)
    
    def make_move(self, pile, trumpsuit, fullGameObject = None):

        if self.strategy == "randomControlled":
            card = self.makeRandomControlledMove(pile, trumpsuit)
        elif self.strategy == "randomUncontrolled":
            card = self.makeRandomUncontrolledMove(pile, trumpsuit)
        elif self.strategy == "basic":
            card = self.makeBasicMove(pile, trumpsuit)
        elif self.strategy == "manualUncontrolled":
            card = self.makeManualUnControlledMove(pile, trumpsuit)
        elif self.strategy =="advanced":
            card = self.makeAdvancedMove(pile, trumpsuit, fullGameObject)
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

    def cardSuitCheck(self, pile, cardPlayed):
         # If the player makes a move that is not the same as is led
        # remove that suit from their possible hand
        if len(pile) > 0:
            if cardPlayed[-1] != pile[0][-1]:
                self.remove_suit(pile[0][-1])
                print "Removing all of suit: ", pile[0][-1]

    def makeRandomControlledMove(self, pile, trumpsuit):
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
        if self.play_card(card):
            self.cardSuitCheck(pile, card)
            return card

    def makeRandomUncontrolledMove(self, pile, trumpsuit):
        print " ------------ Player: ", self.playerNumber, " -----------"
        print "Trumps: ",
        print trumpsuit
        print "All Possible Moves: ",
        print self.possiblehand
        print "Cards in pile: ",
        print pile
        if type(self.possiblehand) is str:
            card = self.possiblehand
        else:
            card = random.choice(self.possiblehand)
        if self.play_card(card):
            self.cardSuitCheck(pile,card)
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
                    self.cardSuitCheck(pile, cardToPlay)
                    return cardToPlay
                else:
                    print "Invalid card choice, try again"
            else:
                print "Invalid card choice, try again"

    def makeManualUnControlledMove(self, pile, trumpsuit):
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
                self.cardSuitCheck(pile, cardToPlay)
                return cardToPlay
            else:
                print "Invalid card choice, try again"

    def makeBasicMove(self, pile, trumpsuit):
        # This makes a basic move based on the cards in hand and the game

        # If I have bid more than I have won then I need to play some higher cards

        return makeRandomMove(pile, trumpsuit)

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
                thisPlayerCardRanking = []
                
                for specificCard in currentPossibleMoves:
                    # We suppress the monte carlo chatter
                    sys.stdout = NullIO()
                    [outputScores, errorCounter] = self.cardPlayMonteCarlo(specificCard,self.monteCarloNumber,fullGameObject)
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
            self.cardSuitCheck(pile,cardToPlay)
            return cardToPlay

    def cardPlayMonteCarlo(self,specificCard,monteCarloNumber,fullGameObject):

        playerRange = range(0, fullGameObject.numberofplayers)
        averageScores = [0.0 for i in playerRange]

        # Keep track of any game errors and use them to correct score estimate
        errorCounter = 0
        for monteIterator in xrange(0,monteCarloNumber): 
            # Make a copy of the current game object
            tempGame = copy.deepcopy( fullGameObject )
            tempGame.convertToDeterministicGame()

            # Reset scores for now
            tempGame.scores = [0.0 for i in playerRange]

            # Revert all players to random
            for p in tempGame.players:
                if p.strategy == "advanced":
                    p.strategy = "randomControlled"
                else:
                    p.strategy = "randomControlled"

            try:
                tempGame.playPartialTrick(specificCard)
                tempGame.playPartialRound()

                # Merge the score output
                averageScores = map(add, averageScores, tempGame.scores)
                #for i in range(0,tempGame.numberofplayers):
                #    averageScores[i] = averageScores[i] + float(tempGame.scores[i])
            except:
                tb = traceback.format_exc()
                errorCounter = errorCounter + 1

        # Convert to proper average
        for i in range(0,tempGame.numberofplayers):
            averageScores[i] = (1/float(monteCarloNumber - errorCounter))*averageScores[i]
        return averageScores, errorCounter

    def pick_trumps(self):
        if self.strategy == "randomControlled":
            thistrump = TrumpPicker.picktrump(self.possiblehand)
        elif self.strategy == "randomUncontrolled":
            thistrump = TrumpPicker.picktrump(self.possiblehand)
        elif self.strategy == "basic":
            thistrump = TrumpPicker.picktrump(self.possiblehand)
        elif self.strategy == "manualUncontrolled":
            thistrump = self.pickManualTrump()
        elif self.strategy =="advanced":
            thistrump = TrumpPicker.picktrump(self.possiblehand)
        return thistrump

    def pickManualTrump(self):
        print "Enter trump: "
        trump = raw_input().rstrip()
        return trump

    def make_bid(self, nplayers, ncards, trumpsuit, previousbids):

        if self.strategy == "randomControlled":
            thisbid = self.makeAverageBid(nplayers, ncards, trumpsuit, previousbids)
        elif self.strategy == "randomUncontrolled":
            thisbid = self.makeAverageBid(nplayers, ncards, trumpsuit, previousbids)
        elif self.strategy == "basic":
            thisbid = self.makeAverageBid(nplayers, ncards, trumpsuit, previousbids)
        elif self.strategy == "manualUncontrolled":
            thisbid = self.makeManualBid()
        elif self.strategy =="advanced":
            thisbid = self.makeAdvancedBid( nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def checkValidBid(self,nplayers,ncards,bid,previousbids):
        if self.canBidZero == False:
            if bid == 0:
                return False
        if len(previousbids) < nplayers-1:
            return True
        else:
            if len(previousbids) == nplayers -1:
                if ( sum(previousbids) + bid )== ncards:
                    return False
                else:
                    return True

    def getValidBidOptions(self,nplayers,ncards,previousbids):
        validBids = []
        for bid in range(0,ncards+1):
            if self.checkValidBid(nplayers,ncards,bid,previousbids):
                validBids.append(bid)
        return validBids
    
    def makeAverageBid(self,nplayers, ncards, trumpsuit, previousbids):
        originalbid = int(round(ncards/float(nplayers)))
        validBids = self.getValidBidOptions(nplayers,ncards,previousbids)
        for i in range(0,ncards):
            thisbid = (originalbid + i) % ncards
            if thisbid in validBids:
                return thisbid

    def makeManualBid(self):
        print "Enter bid: "
        thisbid = int(raw_input())
        return thisbid
    
    def makeAdvancedBid(self, nplayers, ncards, trumpsuit, previousbids):
        cardsInHand = self.possiblehand
        vicProbList = self.computeProbability1CardVictory(nplayers, cardsInHand, ncards, trumpsuit)
        bidPDF = monte_carlo_pdfify(vicProbList,5000)

        validBids = self.getValidBidOptions(nplayers,ncards,previousbids)
        for i in range(0,ncards):
            if i not in validBids:
                bidPDF[i] = -1.0

        [confidence, thisbid] = max_and_index(bidPDF)
        print "Bid: ", thisbid
        print "Bid confidence: ", confidence
        return thisbid

    def computeProbability1CardVictory(self, nplayers, cardsInHand, ncards, trumpsuit):
        outputProbs = []
        handSize = len(cardsInHand)
        for card in cardsInHand:
            # All cards that beat this card, assuming this card leads
            cardlist = card_beating_list(card, trumpsuit)
            # All cards that beat this card and are not in my hand
            possibleOppList = list( set(cardsInHand)^set(cardlist) )
            # Probability that an opposition player does not have one of these cards
            probabilityOfWin1v1 = 1 - ( len(  possibleOppList  ) / float(52 - handSize) )
            # I know this is wrong, but not very wrong, needs some sampling without replacement stuff...
            # Probability that no opposition players have one of these cards
            probabilityTotalVictory = probabilityOfWin1v1**( nplayers -1 )
            outputProbs.append( probabilityTotalVictory )
        return outputProbs

    def advanceZeroCounter(self,bid):
        if bid == 0:
            print "Zero counter: ", self.zeroCounter
            self.zeroCounter = self.zeroCounter + 1
            if self.zeroCounter >= 2:
                self.canBidZero = False
        else:
            self.zeroCounter = 0
            self.canBidZero = True

    def convertToValidHand(self):
        print self.possiblehand, self.cardsLeftInHand
        try:
            newHand = random.sample(self.possiblehand,self.cardsLeftInHand)
        except:
            newHand = []
            print "error"
        return newHand

    def setPossibleHand(self,newHand):
        if len( newHand ) < self.cardsLeftInHand:
            a = 1
            print "This is the bad bit"
        self.possiblehand = [i for i in newHand]

        
