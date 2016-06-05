


class Player():
	def __init__(self, specificcards = []):
		if not specificcards:
			self.possiblehand = [ str(x) + 'd' for x in range( 2, 14 )]
			self.possiblehand += [ str(x) + 'h' for x in range( 2, 14 )]
			self.possiblehand += [ str(x) + 'c' for x in range( 2, 14 )]
			self.possiblehand += [ str(x) + 's' for x in range( 2, 14 )]
		else:
			self.possiblehand = specificcards

	def remove_possible(card):
		if card in self.possiblehand:
			self.possiblehand.remove(card)


class Game():
	def __init__(thisplayercards):

		# Game related bant
		self.numberofplayers = 2
		self.round = 0
		self.cardsplayed = 0
		self.dealer = 0
		self.trumps = 'h'
		self.players = []

		# The player we play as
		self.players.append(new Player )

		# The other players

