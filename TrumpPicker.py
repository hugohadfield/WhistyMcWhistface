import numpy
import random

def picktrump(hand):

    #Process shit
    handsize = len(hand)
    trump = 0
    h = 0
    hearts = []
    diamonds = []
    clubs = []
    spades = []    
    for card in hand:
        if len(card)==2:
            number = int(card[0])
        else:
            number = 10 + int(card[1])
        suit = card[-1]
        if suit == 'h':
            hearts.append(number)
        elif suit == 'd':
            diamonds.append(number)
        elif suit == 'c':
            clubs.append(number)
        elif suit == 's':
            spades.append(number)
        else:
            print("Error: card does not have a suit...")
            
    #Stats for each suit
    mini=handsize
    maxi=0
    maxav=0
    minav=14
    high=0
    suits = [] # Holds cards from one suit
    for suit in [hearts,diamonds,clubs,spades]:
        suits.append(suit)
        suitsize = len(suit)
        
        if suitsize<mini:
            mini=suitsize
            minisuit=suit
        if suitsize>maxi:
            maxi=suitsize
            maxisuit=suit
        av = numpy.sum(suit)/suitsize
        if av>maxav:
            maxav=av
            maxavsuit=suit
        if av<minav:
            minav=av
            minavsuit=suit
        if (suitsize>0):
            h = max(suit)
        if h>high:
            high=h
            highsuit=suit
        
    #Stats for whole hand
    numdist = hearts + diamonds + clubs + spades
    numdist.sort()
    c = numpy.mean(numdist)
    ss = sum((x-c)**2 for x in numdist)
    var = ss/handsize
    sdev = var**0.5
    
    #Pick stance
    if (c<6.6) & (sdev<3):
        stance = 'donald'
    elif (handsize>1) & (maxi>handsize*0.49) & (numpy.mean(maxisuit)>8):
        stance = 'aggressive'
    else:
        stance = 'safe'
    
    #Pick trump
    if stance == 'donald':
        if mini==0:
            trump = minisuit
        elif minisuit==minavsuit:
            trump = minisuit
        elif len(minavsuit) < 0.5*handsize:
            trump = minavsuit
        else:
            trump = minisuit
    elif stance == 'aggressive':
        trump = maxisuit
    elif stance == 'safe':
        if numpy.mean(maxisuit)>9.9:
            trump = maxisuit
        elif highsuit == maxisuit:
            trump = highsuit
        else:
            trump = maxavsuit
    
    #Return trump
    if trump==hearts:
        trump='h'
    elif trump==diamonds:
        trump='d'
    elif trump==clubs:
        trump = 'c'
    elif trump==spades:
        trump = 's'
    else:
        print("Something has gone wrong picking the trump.")
    return trump

def pickRandomTrump():
    return random.choice(["h", "c", "d", "s"])
