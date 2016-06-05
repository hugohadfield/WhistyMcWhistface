
import copy
from WhistLib import *
from WhistPlayer import *

class Game():
    def __init__(self):
        # Game related variables
        self.numberofplayers = 3
        self.cardnumbers = [7,6,5,4,3,2,1,2,3,4,5,6,7]
        self.players = []
        self.resetGame()

    def playRound(self):
        # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        
        # Deal all the cards
        self.dealAllCards(cardsinround)
        
        # Reset the game variables
        self.resetRound()

        # The dealer picks trumps
        self.trumpsuit = self.players[self.dealer].pick_trumps()
        
        # Get all the bids
        self.getPlayerBids()
        
        # Play all the tricks
        self.turn = (self.dealer + 1)% self.numberofplayers
        for trick in range(0,cardsinround):
            self.playFullTrick()

        self.processScore()
        print(self.scores)
        self.endRound()
            
    def endRound(self):
        # Increment the round number and the dealer number
        self.roundnumber = self.roundnumber + 1
        self.dealer = (self.dealer + 1) % self.numberofplayers

    def extractTrickWinner(self):
        winningplay = analyse_pile(self.pile, self.trumpsuit)
        return winningplay

    def updateTricksWon(self, winningplay):
        winningPlayerIndex = (self.dealer + 1 + winningplay)% self.numberofplayers
        self.tricksWon[winningPlayerIndex] = self.tricksWon[winningPlayerIndex] + 1

    def incrementTurn(self):
        self.turn = ( self.turn + 1) % self.numberofplayers

    def getPlayerBids(self):
        # Let the players make their bid
        bidno = self.dealer
        for n in range(0,self.numberofplayers):
            bidno = (bidno + 1) % self.numberofplayers
            player = self.players[bidno]
            self.bids[bidno] = player.make_bid( self.trumpsuit )

    def resetRound(self):
         # Empty the tricks and the bids
        self.bids = []
        self.tricksWon = []
        for i in range(0,self.numberofplayers):
            self.bids.append(-1)
            self.tricksWon.append(0)

    def resetGame(self):
        # Transient per round
        self.pile = []
        self.bids = []
        self.scores = []
        for pIt in range(0,self.numberofplayers):
            self.scores.append(0)
        self.dealer = 0
        self.trumpsuit = 'h'
        self.roundnumber = 0

    def dealAllCards(self,cardsinround):
         # Deal the cards
        playerhands = generate_random_deal(cardsinround,self.numberofplayers)
        self.players = []
        for hand in playerhands:
            thisplayer = Player( hand )
            self.players.append(copy.deepcopy(thisplayer))

    def playFullTrick(self):
        self.pile = []
        for n in range(0,self.numberofplayers):
            player = self.players[ self.turn ]
            # Each player makes their move
            self.pile.append( player.make_move( self.pile) )
            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)

    def tricksToScore(self,bid,trickNo):
        playerScore = trickNo
        if ( bid == trickNo ):
            playerScore = playerScore + 10
        return playerScore

    def processScore(self):
        # Compare the number of tricks that each player has won 
        # with the number of tricks that they bid to win
        for playerIterator in range(0,self.numberofplayers):
            thisTrickCount = self.tricksWon[playerIterator]
            thisBid = self.bids[playerIterator]
            thisPlayerScore = self.tricksToScore(thisBid,thisTrickCount)
            self.scores[playerIterator] = thisPlayerScore




