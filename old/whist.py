
def max_and_index(listin):
	index, value = max(enumerate(my_list), key=operator.itemgetter(1))
	return index, value


class Player():
	def __init__(self):
		self.possiblehand = [ str(x) + 'D' for x in range( 2, 14 )]
		self.possiblehand += [ str(x) + 'H' for x in range( 2, 14 )]
		self.possiblehand += [ str(x) + 'C' for x in range( 2, 14 )]
		self.possiblehand += [ str(x) + 'S' for x in range( 2, 14 )]

	def remove_possible(card):
		if card in self.possiblehand:
			self.possiblehand.remove(card)


def intersect(a, b):
    return list(set(a) & set(b))

def card_beating_list(card, trumpsuit):
	numberlist = range( int(card[0])+1, 14 )
	cardlist = [ str(x) + card[1] for x in numberlist]
	if card[1] != trumpsuit:
		alltrumps = [ str(x) + trumpsuit for x in range( 2, 14 )]
		cardlist = cardlist + alltrumps
	return cardlist

def player_beating_list(cardlist,player):
	return intersect(player.possiblehand, cardlist)

def player_beating_probability(card,trumpsuit,player):
	numer = len( player_beating_list(card_beating_list(card, trumpsuit) , player) )
	denom = len( player.possiblehand )
	return float(numer)/denom


card = '5H'
trumpsuit = 'H'
player1 = Player();

print player_beating_probability(card,trumpsuit,player1)


def possible_trick_win(pile,trumpsuit,player):
	# Could a specific player win this trick?
	for card in pile:
		pilewin = False
		if player_beating_probability(card,trumpsuit,player) > 0.00001:
			pilewin = True
			break
	return pilewin


# The case that a player is aiming to win the trick and they are second last to play
if n = N-1 and b = >0:
	probabilitylist = []
	for card in hand:
		# For each card in your hand see what the probability of the next player being able to beat you is
		probabilitylist.append( player_beating_probability(card,trumpsuit,playerlist[n]) )
	return max_and_index(probabilitylist)


# Third last player:
Can I win this trick?
For each of the cards in the next persons possiblehand, what is the probability that the last person will win




