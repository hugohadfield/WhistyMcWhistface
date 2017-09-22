
import copy
from WhistLib import *
import TrumpPicker
import sys
import traceback
from operator import add

class Player():
    def __init__(self, playerNumber, specificCards = [], additionalParameters=[]):
        self.playerNumber = playerNumber
        self.resetZeroCounter()
        self.cardsLeftInHand = 7
        if not specificCards:
            self.possibleHand = [i for i in self.getListOfAllPossibleHands()]
        else:
            self.possibleHand = [i for i in specificCards]
        if additionalParameters:
            self.handleAdditionalParamaters(additionalParameters)
        self.realHand = [i for i in specificCards]

    def handleAdditionalParamaters(self,parameters):
        return True

    def resetZeroCounter(self):
        self.canBidZero = True
        self.zeroCounter = 0  

    def getListOfAllPossibleHands(self):
        thisList = [ str(x) + 'h' for x in range( 2, 15 )]
        thisList += [ str(x) + 'd' for x in range( 2, 15 )]
        thisList += [ str(x) + 'c' for x in range( 2, 15 )]
        thisList += [ str(x) + 's' for x in range( 2, 15 )]
        return thisList

    def removePossible(self, cards):
        try:
            newHand = [i for i in self.possibleHand]
            if type(cards) is str:
                cleanCard = cards.rstrip()
                if cleanCard in newHand:
                    newHand.remove(cleanCard)
                    self.setPossibleHand(newHand)
            else:
                for card in cards:
                    cleanCard = card.rstrip()
                    if cleanCard in newHand:
                        newHand.remove(cleanCard)
                        self.setPossibleHand(newHand)
        except:
            print "Card removal error"

    def removeReal(self, cards):
        newHand = [i for i in self.realHand]
        if type(cards) is str:
            cleanCard = cards.rstrip()
            if cleanCard in newHand:
                newHand.remove(cleanCard)
                self.setRealHand(newHand)
        else:
            for card in cards:
                cleanCard = card.rstrip()
                if cleanCard in newHand:
                    newHand.remove(cleanCard)
                    self.setRealHand(newHand)
                
    def playCard(self, card):
        cleanCard = card.rstrip()
        if cleanCard in self.possibleHand:
            print ["Removing: ", cleanCard]
            self.cardsLeftInHand = self.cardsLeftInHand - 1
            self.removePossible(cleanCard)
            self.removeReal(cleanCard)
            return True
        else:
            print " Card is not in hand "
            print cleanCard
            print self.possibleHand
            return False

    def removeSuit(self, card):
        suit = card[-1]
        cardsToRemove = cards_of_suit(self.possibleHand, suit)
        self.removePossible(cardsToRemove)

    def getPossibleMoves(self,pile):
        if len(pile) > 0:
            ledCards = cards_of_suit(self.possibleHand, pile[0][-1])
            if len(ledCards) > 0:
                return ledCards
            else:
                return self.possibleHand
        else:
            return self.possibleHand

    def getRealPossibleMoves(self,pile):
        if len(pile) > 0:
            ledCards = cards_of_suit(self.realHand, pile[0][-1])
            if len(ledCards) > 0:
                return ledCards
            else:
                return self.realHand
        else:
            return self.realHand

    def cardSuitCheck(self, pile, cardPlayed):
         # If the player makes a move that is not the same as is led
        # remove that suit from their possible hand
        if len(pile) > 0:
            if cardPlayed[-1] != pile[0][-1]:
                self.removeSuit(pile[0][-1])
                print "Removing all of suit: ", pile[0][-1]

    def pick_trumps(self, nplayers):
        thistrump = TrumpPicker.picktrump(self.realHand)
        return thistrump

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
        print self.possibleHand, self.cardsLeftInHand
        try:
            newHand = random.sample(self.possibleHand, self.cardsLeftInHand)
        except:
            newHand = []
            print "error"
        return newHand

    def setPossibleHand(self,newHand):
        if len( newHand ) < self.cardsLeftInHand:
            a = 1
            print "This is the bad bit"
        self.possibleHand = [i for i in newHand]

    def setRealHand(self,newHand):
        self.realHand = [i for i in newHand]
