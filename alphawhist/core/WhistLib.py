import random
import io
import operator
from operator import mul

GRAPHICAL_INPUT = False
try:
    import easygui

    GRAPHICAL_INPUT = True
except:
    print('easygui not available, defaulting to text input')


def tricksToScore(bid, trickNo):
    playerScore = trickNo
    if ( bid == trickNo ):
        playerScore = playerScore + 10
    return playerScore

def gen_all_cards_in_deck():
    numberlist = list(range(2, 15))
    all_cards = []
    for i in ['h', 'c', 's', 'd']:
        for x in numberlist:
            all_cards.append(str(x) + i)
    return all_cards

all_cards_in_deck = gen_all_cards_in_deck()

def gen_card_to_index_mapping():
    return {c: i for i,c in enumerate(all_cards_in_deck)}


card_to_index_map = gen_card_to_index_mapping()
index_to_card_map = {v: k for k, v in card_to_index_map.items()}


inv_card_name_mapping = {'Js': '11s', 'Jc': '11c', 'Jh': '11h', 'Jd': '11d',
                         'Qs': '12s', 'Qc': '12c', 'Qh': '12h', 'Qd': '12d',
                         'Ks': '13s', 'Kc': '13c', 'Kh': '13h', 'Kd': '13d',
                         'As': '14s', 'Ac': '14c', 'Ah': '14h', 'Ad': '14d'}

card_name_mapping = {v: k for k, v in list(inv_card_name_mapping.items())}


def translate_card_name(card_name):
    graphics_name = card_name_mapping.get(card_name, card_name)
    return graphics_name


def translate_inv_card_name(card_name):
    internal_name = inv_card_name_mapping.get(card_name, card_name)
    return internal_name


def generate_easygui_deal(cardsinround):
    areYouSure = False
    while not areYouSure:
        translated_choices = [translate_card_name(i) for i in all_cards_in_deck]
        hand = easygui.multchoicebox(msg='Select ' + str(cardsinround) + ' cards', title='Dealing',
                                     choices=translated_choices)
        translated_hand = [translate_inv_card_name(i) for i in hand]
        if len(translated_hand) == cardsinround:
            areYouSure = True
    return translated_hand


def generate_cmd_deal(cardsinround):
    areYouSure = False
    while not areYouSure:
        print("Please enter cards, 1 at a time: ")
        hand = []
        for cIt in range(0, cardsinround):
            card = input().rstrip()
            hand.append(card)
        print(hand)
        print("Are you sure? (y/n)")
        if input().rstrip() == "y":
            areYouSure = True
    return hand


def generate_manual_deal(cardsinround):
    if GRAPHICAL_INPUT:
        return generate_easygui_deal(cardsinround)
    else:
        return generate_cmd_deal(cardsinround)


class NullIO(io.StringIO):
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
                if len(card) == 2:
                    number = int(card[0])
                else:
                    number = 10 + int(card[1])
                numbers.append(number)
            winningcard = leadlist[numbers.index(max(numbers))]
    else:
        # Find which is the best
        for card in trumplist:
            if len(card) == 2:
                number = int(card[0])
            else:
                number = 10 + int(card[1])
            numbers.append(number)
        winningcard = trumplist[numbers.index(max(numbers))]
    return pile.index(winningcard)


def generate_random_hand(ncards):
    thishand = []
    k = 0
    while k < ncards:
        num = random.randint(2, 14)
        suit = random.randint(1, 4)
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
            k = k + 1
    return thishand


def generate_random_deal(ncards, nplayers):
    if ncards * nplayers > 52:
        print("Error: the requested deal contains over 52 cards.")
        return 0
    cards = []
    hands = []
    k = 0
    while k < ncards * nplayers:
        num = random.randint(2, 14)
        suit = random.randint(1, 4)
        if suit == 1:
            card = str(num) + 'h'
        elif suit == 2:
            card = str(num) + 'd'
        elif suit == 3:
            card = str(num) + 'c'
        else: #if suit == 4:
            card = str(num) + 's'
        if not card in cards:
            cards.append(card)
            k = k + 1
    # print(cards)
    for i in range(0, nplayers):
        hands.append(cards[(i * ncards):((i + 1) * ncards)])
    return hands


def cards_of_suit(cardlist, suit):
    outputlist = []
    for thiscard in cardlist:
        if thiscard[-1] == suit:
            outputlist.append(thiscard)
    return outputlist


def card_beating_list(card, trumpsuit):
    # This is a list of all the possible cards that beat a specific card
    # All cards of the same suit that are larger than it and all trumps 
    # if it is not a trump
    numberlist = list(range(int(card[:-1]) + 1, 15))
    cardlist = [str(x) + card[-1] for x in numberlist]
    if card[-1] != trumpsuit:
        alltrumps = [str(x) + trumpsuit for x in range(2, 15)]
        cardlist = cardlist + alltrumps
    return cardlist


def card_following_list(card, leadsuit, trumpsuit):
    # A trump can only be beaten by a higher trump, not by the original lead suit.
    if card[-1] == trumpsuit:
        return card_beating_list(card, trumpsuit)
    # This is a list of all the possible cards that beat a specific card assuming that it is not leading
    # All cards of the lead suit, all cards that are larger than it of the lead suit and all trumps
    # if it is not a trump
    cardlist = []
    if card[-1] == leadsuit:
        numberlist = list(range(int(card[:-1]) + 1, 15))
        cardlist = [str(x) + card[-1] for x in numberlist]
    else:
        if card[-1] != trumpsuit and leadsuit != trumpsuit:
            alllead = [str(x) + leadsuit for x in range(2, 15)]
            cardlist = cardlist + alllead
    if card[-1] != trumpsuit:
        alltrumps = [str(x) + trumpsuit for x in range(2, 15)]
        cardlist = cardlist + alltrumps
    return cardlist


def player_beating_list(card, trumpsuit, player):
    # This is a list of the possible cards that a specific player can play that would beat your card
    cardlist = card_beating_list(card, trumpsuit)
    return intersect(player.possible_hand, cardlist)


def player_beating_probability(card, trumpsuit, player):
    # This is the probability that a specific player can beat your card
    numer = len(player_beating_list(card, trumpsuit, player))
    denom = len(player.possible_hand)
    return float(numer) / denom if denom else 0.0


def list_all_possible_player_cards(allplayers):
    allplayercards = []
    for thisp in allplayers:
        allplayercards = union(allplayercards, thisp.possible_hand)
    return allplayercards


def all_players_beating_list(card, trumpsuit, allplayers):
    # This returns a list of all of the cards that the opposition might hold that could beat your card
    cardlist = card_beating_list(card, trumpsuit)
    allplayercards = list_all_possible_player_cards(allplayers)
    return intersect(cardlist, allplayercards)


def all_players_beating_probability(card, trumpsuit, oppositionplayers):
    # This is the probability that any one of the opposition has a card that will beat yours
    # TODO double check this maths...
    # Also the probability of losing the last trick given that you are leading it
    oppositionbeatingcards = all_players_beating_list(card, trumpsuit, oppositionplayers)
    alloppositioncards = list_all_possible_player_cards(oppositionplayers)
    numer = len(oppositionbeatingcards)
    denom = len(alloppositioncards)
    return float(numer) / denom if denom else 0.0


def probability_player_leads_suit(player, suit):
    # This is the probability that a specific player leads with a specific suit for the case they have one card
    numer = len(cards_of_suit(player.possible_hand, suit))
    denom = len(player.possible_hand)
    return float(numer) / denom if denom else 0.0


def max_and_index(listin):
    index, value = max(enumerate(listin), key=operator.itemgetter(1))
    return value, index


def monte_carlo_pdfify(problist, mcnumber):
    ncards = len(problist)
    pdfout = []
    for i in range(0, ncards + 1):
        pdfout.append(0.0)
    for mcIterator in range(0, mcnumber):
        accumulator = 0
        for cardIt in range(0, ncards):
            if random.random() < problist[cardIt]:
                accumulator = accumulator + 1
        pdfout[accumulator] = pdfout[accumulator] + 1
    for i in range(0, ncards + 1):
        pdfout[i] = pdfout[i] / float(mcnumber)
    return pdfout


def mini_monte_sim(leadingProbList, followingProbList, bidPosition, mcnumber):
    """Approximate trick-count distribution with a changing leader.

    After winning we lead next; after losing another player leads.
    """
    if len(leadingProbList) != len(followingProbList):
        raise ValueError("Lead and follow probability lists must have equal lengths")
    if mcnumber < 1:
        raise ValueError("mcnumber must be positive")
    n = len(leadingProbList)
    pdf = [0.0] * (n + 1)
    for _ in range(mcnumber):
        leading = bidPosition == 0
        wins = 0
        lead = list(leadingProbList)
        follow = list(followingProbList)
        for _ in range(n):
            index = random.randrange(len(lead))
            p_lead, p_follow = lead.pop(index), follow.pop(index)
            won = random.random() < (p_lead if leading else p_follow)
            wins += won
            leading = won
        pdf[wins] += 1
    return [count / mcnumber for count in pdf]

def mini_monte_reward_function(cardIndex, pile, leadingProbList, followingProbList, bidPosition, mcnumber):
    copiedLeadList = [i for i in leadingProbList]
    leadLoss = copiedLeadList.pop(cardIndex)

    copiedFollowList = [i for i in followingProbList]
    followLoss = copiedFollowList.pop(cardIndex)
    # mini_monte_sim(, followingProbList, bidPosition, mcnumber)
    # mini_monte_sim()

    ncards = len(leadingProbList)
    pdfout = []
    for i in range(0, ncards + 1):
        pdfout.append(0.0)

    if bidPosition == 0:
        startleading = True
    else:
        startleading = False
    for mcIterator in range(0, mcnumber):
        leading = startleading
        accumulator = 0
        leadCopy = [i for i in leadingProbList]
        followCopy = [i for i in followingProbList]
        for cardIterator in range(0, ncards):
            cardIndex = random.choice(list(range(0, len(leadCopy))))
            leadProb = leadCopy.pop(cardIndex)
            followProb = followCopy.pop(cardIndex)
            randomVal = random.random()
            if leading:
                if randomVal < leadProb:
                    accumulator = accumulator + 1
            else:
                if randomVal < followProb:
                    accumulator = accumulator + 1
        pdfout[accumulator] = pdfout[accumulator] + 1
    for i in range(0, ncards + 1):
        pdfout[i] = pdfout[i] / float(mcnumber)
    return pdfout

def estimate_card_win_probabilities(cards_in_hand, nplayers, ncards, trumpsuit, samples=512):
    """Monte Carlo single-trick win rates, conditional on lead/follow status.

    Opponents receive non-overlapping uniform random unseen cards and select
    uniformly among legal plays (following suit if possible). Following
    trials condition on our candidate card being legal for the sampled lead.
    This is a random-opponent approximation, not a calibrated strategic model.
    """
    hand = list(cards_in_hand)
    if not hand:
        return [], []
    if nplayers < 2 or ncards < len(hand) or ncards < 1:
        raise ValueError("Invalid number of players or cards")
    if len(set(hand)) != len(hand) or not set(hand).issubset(all_cards_in_deck):
        raise ValueError("Cards must be distinct and from the deck")
    if trumpsuit not in ('h', 'c', 's', 'd') or samples < 1:
        raise ValueError("Invalid trump or sample count")
    unseen = [c for c in all_cards_in_deck if c not in hand]
    total = (nplayers - 1) * ncards
    if total > len(unseen):
        raise ValueError("Not enough unseen cards for opponents")

    leading_wins = [0] * len(hand)
    following_wins = [0] * len(hand)
    following_attempts = [0] * len(hand)

    def legal_random_play(remaining_hand, led_suit):
        matching = cards_of_suit(remaining_hand, led_suit)
        return random.choice(matching if matching else remaining_hand)

    for _ in range(samples):
        dealt = random.sample(unseen, total)
        opponents = [dealt[i * ncards:(i + 1) * ncards]
                     for i in range(nplayers - 1)]

        for i, card in enumerate(hand):
            trick = [card] + [legal_random_play(opp, card[-1])
                              for opp in opponents]
            leading_wins[i] += analyse_pile(trick, trumpsuit) == 0

        our_position = random.randrange(1, nplayers)
        led_card = random.choice(opponents[0])
        prefix = [led_card]
        for opp in opponents[1:our_position]:
            prefix.append(legal_random_play(opp, led_card[-1]))
        held_lead_suit = any(c[-1] == led_card[-1] for c in hand)

        for i, card in enumerate(hand):
            if held_lead_suit and card[-1] != led_card[-1]:
                continue
            following_attempts[i] += 1
            trick = prefix + [card] + [
                legal_random_play(opp, led_card[-1])
                for opp in opponents[our_position:]
            ]
            following_wins[i] += analyse_pile(trick, trumpsuit) == our_position

    lead_probs = [n / samples for n in leading_wins]
    follow_probs = [wins / attempts if attempts else 0.0
                    for wins, attempts in zip(following_wins, following_attempts)]
    return lead_probs, follow_probs
