
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

    def playFullRound(self):
        # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        
        # Deal all the cards
        strategyList = ["manual","random","random"]
        self.dealAllCards(cardsinround,strategyList)

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
            self.turnProgress = trick

        self.processScore()
        print(self.scores)
        self.endRound()

    def playPartialRound(self):
         # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        
        # Play the remaining tricks
        for trick in range(self.turnProgress,cardsinround):
            self.playFullTrick()
            self.turnProgress = trick

        self.processScore()
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
        self.turnProgress = self.turnProgress + 1
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
        self.turnProgress = 0
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
        self.turnProgress = 0

    def dealAllCards(self,cardsinround,strategyList):
        # Deal the cards
        playerhands = generate_random_deal(cardsinround,self.numberofplayers)
        self.players = []
        playerNumber = 0
        for hand in playerhands:
            thisplayer = Player( playerNumber, hand, strategyList[playerNumber])
            self.players.append(copy.deepcopy(thisplayer))
            playerNumber = playerNumber + 1

    def playFullTrick(self):
        self.turnProgress = 0
        self.pile = []
        for n in range(0,self.numberofplayers):
            player = self.players[ self.turn ]
            # Each player makes their move
            cardPlayed = player.make_move( self.pile, self.trumpsuit)
            self.pile.append( cardPlayed )
            print "Player: ", n, " played: ",
            print cardPlayed
            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)

    def playPartialTrick(self,specificCard):
        player = self.players[ self.turn ]
        if player.play_card(specificCard):
            self.pile.append(specificCard)
            self.incrementTurn()
        else:
            print "Error"

        startingPoint = self.turnProgress
        for n in range(startingPoint,self.numberofplayers):
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




