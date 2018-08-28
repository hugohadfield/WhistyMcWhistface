from alphawhist.core.WhistPlayer import *
import sys
import json

class Game():
    def __init__(self, playerConfigFileName):
        # Game related variables
        self.cardnumbers = [7,6,5,4,3,2,1,2,3,4,5,6,7]
        self.playerConfigFileName = ""
        self.playerConfigFileName = playerConfigFileName
        self.numberofplayers = 0
        self.players = []
        self.generatePlayersFromConfigFile(self.playerConfigFileName)
        self.resetGame()

    def parseGameConfigFile(self, configFileName):
        with open(configFileName) as data_file:
            jsonFileData = json.load(data_file)
            for parameterCollection in jsonFileData["GameParameters"]:
                playerConfigName = parameterCollection["PlayerConfigFile"]
                self.playerConfigFileName = playerConfigName

    def generatePlayersFromConfigFile(self,configFileName):
        with open(configFileName) as data_file:
            jsonFileData = json.load(data_file)
            for player in jsonFileData["Players"]:
                playerClassName = player["class"]
                playerNumber = player["playerNumber"]

                module = __import__(playerClassName)
                class_ = getattr(module, playerClassName)
                instance = class_(int(playerNumber), additionalParameters=player["additionalParameters"])
                self.players.append(instance)
        self.numberofplayers = len(self.players)

    def generatePlayersFromList(self, playerClassNameList):
        n = 0
        for playerClassName in playerClassNameList:
            module = __import__(playerClassName)
            class_ = getattr(module, playerClassName)
            instance = class_(n)
            self.players.append(instance)
            n = n + 1

    def playFullGame(self):
        for i in range(0, len(self.cardnumbers)):
            print(("Round number: ", i + 1))
            self.playFullRound()

    def playFullRound(self):
        # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        print(" ")
        print(" ")
        print("------------------------------------------------")
        print("Cards in round: ", end=' ')
        print(cardsinround)

        # Reset the game variables
        self.resetRound()
        
        # Deal all the cards
        self.manualDeal(cardsinround)

        # The dealer picks trumps
        self.trumpsuit = self.players[self.dealer].pick_trumps(self.numberofplayers)
        print("Trumps: ", self.trumpsuit)
        
        # Get all the bids
        self.getPlayerBids()
        
        # Play all the tricks
        self.turn = self.leader
        for trick in range(0,cardsinround):
            self.playFullTrick()

        self.processScore()
        print((self.scores))
        self.endRound()

    def playPartialRound(self):
         # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        print("Cards in round: ", end=' ')
        print(cardsinround)
        
        # Play the remaining tricks
        startingTurn = copy.deepcopy( self.trickNumber )
        for trick in range(startingTurn,cardsinround):
            self.playFullTrick()

        self.processScore()
        print((self.scores))
        self.endRound()


    def playFullTrick(self):
        print(" ")
        print(" ")
        print("########### Beggining full trick number, " , self.trickNumber ,"############")
        print("Bids: ", self.bids)
        print("Player cards ")
        for p in self.players:
            print(p.possible_hand)
        print("#############################################")
        self.pile = []
        for n in range(0,self.numberofplayers):
            player = self.players[ self.turn ]

            # Player makes their move
            cardPlayed = player.makeMove(self.pile, self.trumpsuit, self)

            self.pile.append( cardPlayed )
            print("Player: ", player.playerNumber, " played: ", end=' ')
            print(cardPlayed)

            self.cleanCardFromPlayers(cardPlayed)

            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)
        print("#############################################")
        self.setTurnToWinner()
        self.trickNumber = self.trickNumber + 1
        print(self.tricksWon)
        print("############### End of trick ################")

    def playPartialTrick(self,specificCard):
        player = self.players[ self.turn ]
        if player.playCard(specificCard):
            self.pile.append(specificCard)
            self.incrementTurn()
        else:
            print("Error")

        print("Player: ", player.playerNumber, " played: ", end=' ')
        print(specificCard)

        self.cleanCardFromPlayers(specificCard)

        # Restart the trick
        startingPoint = len(self.pile)
        for n in range(startingPoint,self.numberofplayers):
            player = self.players[ self.turn ]
            # Each player makes their move
            
            # Player makes their move
            cardPlayed = player.makeMove(self.pile, self.trumpsuit, self)


            self.pile.append( cardPlayed )

            print("Player: ", player.playerNumber, " played: ", end=' ')
            print(cardPlayed)

            self.cleanCardFromPlayers(cardPlayed)
            
            self.incrementTurn()

        # See who has won and move on
        self.updateTricksWon(self.extractTrickWinner())
        print("#############################################")
        self.setTurnToWinner()
        self.trickNumber = self.trickNumber + 1
        print(self.tricksWon)
        print("############### End of trick ################")

    def cleanCardFromPlayers(self, card, playerToIgnore = 0):
        for playerTag in self.players:
            if playerTag.playerNumber != playerToIgnore:
                playerTag.removePossible(card)

    def setTurnToWinner(self):
        winningplay = self.extractTrickWinner()
        winningPlayerIndex = (self.leader + winningplay)% self.numberofplayers
        print("Player: ", winningPlayerIndex, " won the trick with so it is now: ", winningPlayerIndex, "'s' go")
        self.leader = winningPlayerIndex
        self.turn = self.leader


    def endRound(self):
        # Increment the round number and the dealer number
        self.trickNumber = 0
        self.roundnumber = self.roundnumber + 1
        self.dealer = (self.dealer + 1) % self.numberofplayers
        self.leader = (self.dealer + 1) % self.numberofplayers

    def extractTrickWinner(self):
        winningplay = analyse_pile(self.pile, self.trumpsuit)
        return winningplay

    def updateTricksWon(self, winningplay):
        winningPlayerIndex = (self.leader + winningplay)% self.numberofplayers
        self.tricksWon[winningPlayerIndex] = self.tricksWon[winningPlayerIndex] + 1

    def incrementTurn(self):
        print("------------------------------------------")
        print("Incrementing turn from ", end=' ')
        print(self.turn, end=' ')
        self.turn = ( self.turn + 1) % self.numberofplayers
        print(" to ", end=' ')
        print(self.turn)
        print("For tick number: ", self.trickNumber)
        
        print("------------------------------------------")

    def getPlayerBids(self):
        # Let the players make their bid
        bidno = self.dealer
        currentBids = []
        for n in range(0,self.numberofplayers):
            bidno = (bidno + 1) % self.numberofplayers
            player = self.players[bidno]
            ncards = self.cardnumbers[self.roundnumber]
            thisPlayerBid = player.makeBid(self.numberofplayers, ncards, self.trumpsuit, currentBids)
            currentBids.append(thisPlayerBid)
            self.bids[bidno] = thisPlayerBid

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
        self.leader = (self.dealer + 1) % self.numberofplayers
        self.trumpsuit = 'h'
        self.roundnumber = 0
        self.trickNumber = 0

    def manualDeal(self,cardsinround):
        # Manually deal the first player
        player1hand = generate_manual_deal(cardsinround)
        playerOne = self.players[0]
        playerOne.possible_hand = [i for i in player1hand]
        playerOne.real_hand = [i for i in player1hand]
        # Now track the possible hands of all other players
        for playerNumber in range(1,self.numberofplayers):
            thisplayer = self.players[playerNumber]
            thisplayer.possible_hand = all_cards_in_deck
            thisplayer.removePossible(playerOne.real_hand)
            thisplayer.real_hand = [i for i in thisplayer.possible_hand]

    def randomDeal(self,cardsinround):
        # Generate the real hands
        playerHands = generate_random_deal(cardsinround,self.numberofplayers)
        playerOne = self.players[0]
        # We know our real hand and so don't have to imagine a possible hand
        playerOne.possible_hand = [i for i in playerHands[0]]
        playerOne.real_hand = [i for i in playerHands[0]]
        # The rest of the players are dealt a real hand and the possible hand 
        # they could hold according to the monte carlo player is tracked
        for playerNumber in range(1,self.numberofplayers):
            thisplayer = self.players[playerNumber]
            thisplayer.possible_hand = all_cards_in_deck
            thisplayer.removePossible(playerHands[0])
            thisplayer.real_hand = [i for i in playerHands[playerNumber]]

    def processScore(self):
        # Compare the number of tricks that each player has won 
        # with the number of tricks that they bid to win
        for playerIterator in range(0,self.numberofplayers):
            thisTrickCount = self.tricksWon[playerIterator]
            thisBid = self.bids[playerIterator]
            thisPlayerScore = tricksToScore(thisBid,thisTrickCount)
            self.scores[playerIterator] = self.scores[playerIterator] + thisPlayerScore

    def convertToDeterministicGame(self):
        for thisPlayer in self.players:
            hand = thisPlayer.convertToValidHand()
            thisPlayer.real_hand = hand
            thisPlayer.possible_hand = hand
            playerToIgnore = thisPlayer.playerNumber
            for card in thisPlayer.possible_hand:
                self.cleanCardFromPlayers(card, playerToIgnore)
