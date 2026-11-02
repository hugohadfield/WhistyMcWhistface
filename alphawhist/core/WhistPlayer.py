from alphawhist.core.WhistLib import *
from alphawhist.core import TrumpPicker


class Player():
    def __init__(self, playerNumber, specificCards=None, additionalParameters=None):
        self.playerNumber = playerNumber
        self.resetZeroCounter()
        # Take ownership of both lists instead of keeping caller-owned hands.
        cards = list(specificCards) if specificCards is not None else []
        self.set_hands(cards, cards if cards else all_cards_in_deck)
        if additionalParameters:
            self.handleAdditionalParamaters(additionalParameters)

    @property
    def possible_hand(self):
        # Return a snapshot; callers should use the setter to change a hand.
        return self._possible_hand.copy()

    @possible_hand.setter
    def possible_hand(self, newHand):
        possible = list(newHand)
        if any(card not in possible for card in self.real_hand):
            raise ValueError('New possible hand does not contain real hand')
        if len(set(possible)) != len(possible):
            raise ValueError('Possible hand contains duplicates')
        self._possible_hand = possible

    @property
    def real_hand(self):
        return self._real_hand.copy()

    @real_hand.setter
    def real_hand(self, newHand):
        real = list(newHand)
        if hasattr(self, '_possible_hand') and not set(real).issubset(self._possible_hand):
            raise ValueError('Possible hand must contain real hand')
        if len(set(real)) != len(real):
            raise ValueError('Real hand contains duplicates')
        self._real_hand = real

    def set_hands(self, real_hand, possible_hand=None):
        """Atomically replace both hands without retaining caller-owned lists."""
        real = list(real_hand)
        possible = list(real if possible_hand is None else possible_hand)
        if not set(real).issubset(possible):
            raise ValueError('Possible hand must contain real hand')
        if len(set(real)) != len(real) or len(set(possible)) != len(possible):
            raise ValueError('Hands must not contain duplicate cards')
        self._real_hand = real
        self._possible_hand = possible

    @property
    def n_cards_in_hand(self):
        return len(self.real_hand)

    @property
    def zero_bid_count(self):
        return self._zero_bid_count

    @property
    def can_bid_zero(self):
        return self._zero_bid_count < 2


    def export_policy_state(self):
        """JSON-compatible persistent memory; subclasses may extend this."""
        result = {
            "policy": type(self).__name__,
            "zero_bid_count": self.zero_bid_count,
        }
        if hasattr(self, "monteCarloNumber"):
            result["monteCarloNumber"] = self.monteCarloNumber
        if hasattr(self, "monteCarloWorkers"):
            result["monteCarloWorkers"] = self.monteCarloWorkers
        return result

    def load_policy_state(self, state):
        """Restore supported persistent fields into a compatible policy.

        Custom algorithms should override this and export_policy_state to
        restore their own versioned memory.
        """
        if state is None:
            return
        self.resetZeroCounter(state["zero_bid_count"])
        if "monteCarloNumber" in state and hasattr(self, "monteCarloNumber"):
            self.monteCarloNumber = state["monteCarloNumber"]
        if "monteCarloWorkers" in state and hasattr(self, "monteCarloWorkers"):
            self.monteCarloWorkers = state["monteCarloWorkers"]

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
        if cleanCard in self.real_hand and cleanCard in self.possible_hand:
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
        if type(bid) is not int or not 0 <= bid <= ncards:
            return False
        if len(previousbids) > nplayers - 1:
            return False
        if not self.can_bid_zero and bid == 0:
            # In a one-card round, if all earlier players bid zero,
            # a zero-banned dealer has NO possible bid: 1 is forbidden
            # by the last-bidder sum rule. Permit this forced zero as
            # the narrow exception so the round remains playable.
            forced_dealer_zero = (
                len(previousbids) == nplayers - 1
                and ncards == 1 and sum(previousbids) == 0)
            if not forced_dealer_zero:
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
        if self.n_cards_in_hand > len(self.possible_hand):
            raise ValueError('Not enough possible cards to sample a hand')
        return random.sample(self.possible_hand, self.n_cards_in_hand)

