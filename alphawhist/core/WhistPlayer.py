from alphawhist.core.WhistLib import *
from alphawhist.core import TrumpPicker


class Player():
    def __init__(self, playerNumber, specificCards=[], additionalParameters=[]):
        self.playerNumber = playerNumber
        self.resetZeroCounter()
        self.real_hand = [i for i in specificCards]
        if not specificCards:
            self.possible_hand = [i for i in all_cards_in_deck]
        else:
            self.possible_hand = [i for i in specificCards]
        if additionalParameters:
            self.handleAdditionalParamaters(additionalParameters)

    @property
    def possible_hand(self):
        return self._possible_hand

    @possible_hand.setter
    def possible_hand(self, newHand):
        self._possible_hand = [i for i in newHand]
        for i in self.real_hand:
            if i not in newHand:
                print('\n\n\n\n\n\n\n')
                print(i)
                print(newHand)
                raise ValueError('New possible hand does not contain real hand')

    @property
    def real_hand(self):
        return self._real_hand

    @real_hand.setter
    def real_hand(self, newHand):
        self._real_hand = [i for i in newHand]

    @property
    def n_cards_in_hand(self):
        return len(self.real_hand)

    @property
    def zero_bid_count(self):
        return self._zero_bid_count

    @property
    def can_bid_zero(self):
        return self._zero_bid_count < 2

    def handleAdditionalParamaters(self, parameters):
        return True

    def resetZeroCounter(self, count=0):
        self._zero_bid_count = count

    def removePossible(self, cards):
        newHand = [i for i in self.possible_hand]
        if type(cards) is str:
            cleanCard = cards.rstrip()
            if cleanCard in newHand:
                newHand.remove(cleanCard)
                self.possible_hand = newHand
        else:
            for card in cards:
                cleanCard = card.rstrip()
                if cleanCard in newHand:
                    newHand.remove(cleanCard)
                    self.possible_hand = newHand

    def removeReal(self, cards):
        newHand = [i for i in self.real_hand]
        if type(cards) is str:
            cleanCard = cards.rstrip()
            if cleanCard in newHand:
                newHand.remove(cleanCard)
                self.real_hand = newHand
        else:
            for card in cards:
                cleanCard = card.rstrip()
                if cleanCard in newHand:
                    newHand.remove(cleanCard)
                    self.real_hand = newHand

    def playCard(self, card):
        cleanCard = card.rstrip()
        if cleanCard in self.possible_hand:
            print(["Removing: ", cleanCard])
            self.removeReal(cleanCard)
            self.removePossible(cleanCard)
            return True
        else:
            print(" Card is not in hand ")
            print(cleanCard)
            print(self.possible_hand)
            return False

    def removeSuit(self, card):
        suit = card[-1]
        cardsToRemove = cards_of_suit(self.possible_hand, suit)
        self.removePossible(cardsToRemove)

    def getPossibleMoves(self, pile):
        if len(pile) > 0:
            ledCards = cards_of_suit(self.possible_hand, pile[0][-1])
            if len(ledCards) > 0:
                return ledCards
            else:
                return self.possible_hand
        else:
            return self.possible_hand

    def getRealPossibleMoves(self, pile):
        if len(pile) > 0:
            ledCards = cards_of_suit(self.real_hand, pile[0][-1])
            if len(ledCards) > 0:
                return ledCards
            else:
                return self.real_hand
        else:
            return self.real_hand

    def cardSuitCheck(self, pile, cardPlayed):
        # If the player makes a move that is not the same as is led
        # remove that suit from their possible hand
        if len(pile) > 0:
            if cardPlayed[-1] != pile[0][-1]:
                self.removeSuit(pile[0][-1])
                print("Removing all of suit: ", pile[0][-1])

    def pick_trumps(self, nplayers):
        thistrump = TrumpPicker.picktrump(self.real_hand)
        return thistrump

    def checkValidBid(self, nplayers, ncards, bid, previousbids):
        if not self.can_bid_zero:
            if bid == 0:
                return False
        if len(previousbids) < nplayers - 1:
            return True
        else:
            if len(previousbids) == nplayers - 1:
                if (sum(previousbids) + bid) == ncards:
                    return False
                else:
                    return True

    def getValidBidOptions(self, nplayers, ncards, previousbids):
        validBids = []
        for bid in range(0, ncards + 1):
            if self.checkValidBid(nplayers, ncards, bid, previousbids):
                validBids.append(bid)
        return validBids

    def advanceZeroCounter(self, bid):
        if bid == 0:
            print("Zero counter: ", self.zero_bid_count)
            self._zero_bid_count = self._zero_bid_count + 1
        else:
            self._zero_bid_count = 0

    def convertToValidHand(self):
        print(self.possible_hand, self.n_cards_in_hand)
        try:
            newHand = random.sample(self.possible_hand, self.n_cards_in_hand)
        except:
            newHand = []
            print("error")
        return newHand

