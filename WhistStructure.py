
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
        print " "
        print " "
        print "------------------------------------------------"
        print "Cards in round: ",
        print cardsinround

        # Reset the game variables
        self.resetRound()
        
        # Deal all the cards
        strategyList = ["advanced","randomUncontrolled","randomUncontrolled"]
        # self.dealAllCards(cardsinround,strategyList)
        self.manualDeal(cardsinround,strategyList)

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

    def playPartialRound(self):
         # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        print "Cards in round: ",
        print cardsinround
        
        # Play the remaining tricks
        startingTurn = copy.deepcopy( self.trickNumber )
        for trick in range(startingTurn,cardsinround):
            self.playFullTrick()

        self.processScore()
        print(self.scores)
        self.endRound()
            
    def endRound(self):
        # Increment the round number and the dealer number
        self.trickNumber = 0
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
        print "------------------------------------------"
        print "############   ", self.trickNumber
        print "############   ", self.turn
        print "------------------------------------------"

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
        self.trickNumber = 0
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
        self.trickNumber = 0

    def dealAllCards(self,cardsinround,strategyList):
        # Deal the cards
        playerhands = generate_random_deal(cardsinround,self.numberofplayers)
        self.players = []
        playerNumber = 0
        for hand in playerhands:
            thisplayer = Player( playerNumber, hand, strategy= strategyList[playerNumber])
            self.players.append(copy.deepcopy(thisplayer))
            playerNumber = playerNumber + 1

    def manualDeal(self,cardsinround,strategyList):
        # Deal the cards for player 1
        player1hand = generate_random_deal(cardsinround,2)[1]
        self.players = [ copy.deepcopy( Player( 0, player1hand, strategy= strategyList[0])) ]

        # Everyone else gets random stuff
        for playerNumber in range(1,self.numberofplayers):
            thisplayer = Player( playerNumber, strategy= strategyList[playerNumber])
            thisplayer.remove_possible(player1hand)
            self.players.append(copy.deepcopy(thisplayer))

    def playFullTrick(self):
        print " "
        print " "
        print "########### Beggining full trick ############"
        print self.bids
        print "#############################################"
        self.pile = []
        for n in range(0,self.numberofplayers):
            player = self.players[ self.turn ]

            # Player makes their move
            if player.strategy != "advanced":
                cardPlayed = player.make_move( self.pile, self.trumpsuit)
            else:
                cardPlayed = player.make_move( self.pile, self.trumpsuit, self)

            self.pile.append( cardPlayed )
            print "Player: ", player.playerNumber, " played: ",
            print cardPlayed

            # If the player makes a move that is not the same as is led
            # remove that suit from their possible hand
            if n != 0:
                if cardPlayed[-1] != self.pile[0][-1]:
                    player.remove_suit(self.pile[0][-1])
                    print "Removing all of suit: ", self.pile[0][-1]

            # If a player plays a card remove it from all other players hands
            for playerTag in self.players:
                playerTag.remove_possible(cardPlayed)

            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)
        print "#############################################"
        self.trickNumber = self.trickNumber + 1
        print self.tricksWon
        print "############### End of trick ################"

    def playPartialTrick(self,specificCard):
        player = self.players[ self.turn ]
        if player.play_card(specificCard):
            self.pile.append(specificCard)
            self.incrementTurn()
        else:
            print "Error"

        print "Player: ", player.playerNumber, " played: ",
        print specificCard

        # If the player makes a move that is not the same as is led
        # remove that suit from their possible hand
        if len(self.pile) > 0:
            if specificCard[-1] != self.pile[0][-1]:
                player.remove_suit(self.pile[0][-1])
                print "Removing all of suit: ", self.pile[0][-1]

        # If a player plays a card remove it from all other players hands
        for playerTag in self.players:
            playerTag.remove_possible(specificCard)

        # Restart the trick
        startingPoint = len(self.pile)
        for n in range(startingPoint,self.numberofplayers):
            player = self.players[ self.turn ]
            # Each player makes their move
            
            # Player makes their move
            if player.strategy != "advanced":
                cardPlayed = player.make_move( self.pile, self.trumpsuit)
            else:
                cardPlayed = player.make_move( self.pile, self.trumpsuit, self)

            self.pile.append( cardPlayed )

            print "Player: ", player.playerNumber, " played: ",
            print cardPlayed

             # If the player makes a move that is not the same as is led
            # remove that suit from their possible hand
            if len(self.pile) > 0:
                if cardPlayed[-1] != self.pile[0][-1]:
                    player.remove_suit(self.pile[0][-1])
                    print "Removing all of suit: ", self.pile[0][-1]

            # If a player plays a card remove it from all other players hands
            for playerTag in self.players:
                playerTag.remove_possible(cardPlayed)

            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)
        print "#############################################"
        self.trickNumber = self.trickNumber + 1
        print self.tricksWon
        print "############### End of trick ################"

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

