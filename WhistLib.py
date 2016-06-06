
import random
import StringIO
import operator

class NullIO(StringIO.StringIO):
    def write(self, txt):
       pass

def intersect(a, b):
    # returns the intersection of two lists
    return list(set(a) & set(b))
    
def union(a, b):
    # returns the union of two lists
    return list(set(a) | set(b))
    
def analyse_pile(pile, trumpsuit):
    # Find all trumps
    trumplist = cards_of_suit(pile, trumpsuit)
    numbers = []
    # If no trumps
    if not trumplist:
        # Find all the cards of the leading suit
        leadsuit = pile[0][-1]
        leadlist = cards_of_suit(pile, leadsuit)
        if len(leadlist) == 1:
            winningcard = leadlist[0]
        else:
            # Find which is the best
            for card in leadlist:
                if len(card)==2:
                    number = int(card[0])
                else:
                    number = 10 + int(card[1])
                numbers.append(number)
            winningcard = leadlist[ numbers.index(max(numbers)) ]
    else:
        # Find which is the best
        for card in trumplist:
            if len(card)==2:
                number = int(card[0])
            else:
                number = 10 + int(card[1])
            numbers.append(number)
        winningcard = trumplist[ numbers.index(max(numbers)) ]
    return pile.index(winningcard)
    
def generate_random_hand(ncards):
    thishand = []
    k = 0
    while k < ncards:
        num = random.randint(2,14)
        suit = random.randint(1,4)
        if suit == 1:
            card = str(num) + 'h'
        elif suit == 2:
            card = str(num) + 'd'
        elif suit == 3:
            card = str(num) + 'c'
        elif suit == 4:
            card = str(num) + 's'
        if not card in thishand:
            thishand.append(card)
            k = k+1
    return thishand
    
def generate_random_deal(ncards,nplayers):
    if ncards*nplayers > 52:
        print("Error: the requested deal contains over 52 cards.")
        return 0;
    cards = []
    hands = []
    k = 0
    while k < ncards*nplayers:
        num = random.randint(2,14)
        suit = random.randint(1,4)
        if suit == 1:
            card = str(num) + 'h'
        elif suit == 2:
            card = str(num) + 'd'
        elif suit == 3:
            card = str(num) + 'c'
        elif suit == 4:
            card = str(num) + 's'
        if not card in cards:
            cards.append(card)
            k = k + 1
    #print(cards)
    for i in range(0,nplayers):
        hands.append(cards[(i*ncards):((i+1)*ncards)])
    return hands

def cards_of_suit(cardlist, suit):
    outputlist = []
    for thiscard in cardlist:
        if thiscard[-1] == suit:
            outputlist.append(thiscard)
    return outputlist

def card_beating_list(card, trumpsuit):
    # This is a list of all the possible cards that beat a specific card
    numberlist = range( int(card[0])+1, 15 )
    cardlist = [ str(x) + card[1] for x in numberlist]
    if card[-1] != trumpsuit:
        alltrumps = [ str(x) + trumpsuit for x in range( 2, 15 )]
        cardlist = cardlist + alltrumps
    return cardlist

def player_beating_list(card, trumpsuit, player):
    # This is a list of the possible cards that a specific player can play that would beat your card
    cardlist = card_beating_list(card, trumpsuit)
    return intersect(player.possiblehand, cardlist)

def player_beating_probability(card, trumpsuit, player):
    # This is the probability that a specific player can beat your card
    numer = len( player_beating_list(card,  trumpsuit , player) )
    denom = len( player.possiblehand )
    return float(numer)/denom
    
def list_all_possible_cards(allplayers):
    allplayercards = []
    for thisp in allplayers:
        allplayercards = union(allplayercards, thisp.possiblehand)
    return allplayercards 

def all_players_beating_list(card,trumpsuit,allplayers):
    # This returns a list of all of the cards that the opposition might hold that could beat your card
    cardlist = card_beating_list(card, trumpsuit)
    allplayercards = list_all_possible_cards(allplayers)
    return intersect(cardlist, allplayercards)

def all_players_beating_probability(card, trumpsuit, oppositionplayers):
    # This is the probability that any one of the opposition has a card that will beat yours
    # TODO double check this maths...
    # Also the probability of losing the last trick given that you are leading it
    oppositionbeatingcards = all_players_beating_list(card,trumpsuit,oppositionplayers)
    alloppositioncards = list_all_possible_cards(oppositionplayers)
    numer = len( oppositionbeatingcards )
    denom = len( alloppositioncards  )
    return float(numer)/denom
    
def probability_player_leads_suit(player, suit):
    # This is the probability that a specific player leads with a specific suit for the case they have one card
    numer = len( cards_of_suit(player.possiblehand, suit) )
    denom = len(player.possiblehand)
    return float(numer)/denom


def max_and_index(listin):
    index, value = max(enumerate(listin), key=operator.itemgetter(1))
    return value, index


# Round specific funcs
def uniform_(card, trumpsuit, oppositionplayers):
    # This is the probability that any one of the opposition has a card that will beat yours
    # TODO double check this maths...
    # Also the probability of losing the last trick given that you are leading it
    oppositionbeatingcards = all_players_beating_list(card,trumpsuit,oppositionplayers)
    alloppositioncards = list_all_possible_cards(oppositionplayers)
    numer = len( oppositionbeatingcards )
    denom = len( alloppositioncards  )
    return float(numer)/denom
