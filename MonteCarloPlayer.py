from WhistPlayer import *
from RandomPlayer import *

class MonteCarloPlayer(Player):

    def __init__(self, playerNumber, specificCards = []):
        Player.__init__(self,playerNumber,specificCards)
        self.monteCarloNumber = 5000

    def pick_trumps(self):
        thistrump = TrumpPicker.picktrump(self.possibleHand)
        return thistrump

    def makeMove(self, pile, trumpsuit, fullGameObject=None):
        currentPossibleMoves = self.getPossibleMoves(pile)
        print "Trumps: ",
        print trumpsuit
        print "All Cards In Hand: ",
        print self.possibleHand
        print "Legal Moves: ",
        print currentPossibleMoves
        print "Cards in pile: ",
        print pile
        if currentPossibleMoves is not str:
            if len(currentPossibleMoves) > 1:
                # Project n steps into the future and evaluate the quality of the move
                thisPlayerCardRanking = []

                for specificCard in currentPossibleMoves:
                    # We suppress the monte carlo chatter
                    sys.stdout = NullIO()
                    [outputScores, errorCounter] = self.cardPlayMonteCarlo(specificCard, self.monteCarloNumber,
                                                                           fullGameObject)
                    sys.stdout = sys.__stdout__
                    thisPlayerCardRanking.append(outputScores[self.playerNumber])
                    print errorCounter,
                print " "
                # Get the maximum value and expected point ranking
                print thisPlayerCardRanking
                [expectedPointGain, cardIndex] = max_and_index(thisPlayerCardRanking)

                # Play the card
                cardToPlay = currentPossibleMoves[cardIndex]
            else:
                cardToPlay = currentPossibleMoves[0]
        else:
            cardToPlay = currentPossibleMoves
        if self.playCard(cardToPlay):
            self.cardSuitCheck(pile, cardToPlay)
            return cardToPlay
        else:
            print "Card playing error"


    def cardPlayMonteCarlo(self, specificCard, monteCarloNumber, fullGameObject):

        playerRange = range(0, fullGameObject.numberofplayers)
        averageScores = [0.0 for i in playerRange]

        # Keep track of any game errors and use them to correct score estimate
        errorCounter = 0
        for monteIterator in xrange(0, monteCarloNumber):
            # Make a copy of the current game object
            tempGame = copy.deepcopy(fullGameObject)
            tempGame.convertToDeterministicGame()

            # Reset scores for now
            tempGame.scores = [0.0 for i in playerRange]

            # Revert all players to random
            newPlayers = []
            for p in tempGame.players:
                newPlayers.append( p.generateModelPlayer() )
            tempGame.players = [i for i in newPlayers]

            try:
                tempGame.playPartialTrick(specificCard)
                tempGame.playPartialRound()

                # Merge the score output
                averageScores = map(add, averageScores, tempGame.scores)
                # for i in range(0,tempGame.numberofplayers):
                #    averageScores[i] = averageScores[i] + float(tempGame.scores[i])
            except:
                tb = traceback.format_exc()
                errorCounter = errorCounter + 1

        # Convert to proper average
        for i in range(0, tempGame.numberofplayers):
            averageScores[i] = (1 / float(monteCarloNumber - errorCounter)) * averageScores[i]

        return averageScores, errorCounter

    def makeBid(self, nplayers, ncards, trumpsuit, previousbids):
        thisbid = self.makeAdvancedBid(nplayers, ncards, trumpsuit, previousbids)
        self.bid = thisbid
        self.advanceZeroCounter(thisbid)
        return thisbid

    def makeAdvancedBid(self, nplayers, ncards, trumpsuit, previousbids):
        cardsInHand = self.possibleHand
        vicProbList = self.computeProbability1CardVictory(nplayers, cardsInHand, ncards, trumpsuit)
        bidPDF = monte_carlo_pdfify(vicProbList, 5000)

        validBids = self.getValidBidOptions(nplayers, ncards, previousbids)
        for i in range(0, ncards):
            if i not in validBids:
                bidPDF[i] = -1.0

        [confidence, thisbid] = max_and_index(bidPDF)
        print "Bid: ", thisbid
        print "Bid confidence: ", confidence
        return thisbid

    def computeProbability1CardVictory(self, nplayers, cardsInHand, ncards, trumpsuit):
        outputProbs = []
        handSize = len(cardsInHand)
        for card in cardsInHand:
            # All cards that beat this card, assuming this card leads
            cardlist = card_beating_list(card, trumpsuit)
            # All cards that beat this card and are not in my hand
            possibleOppList = list(set(cardsInHand) ^ set(cardlist))
            # Probability that an opposition player does not have one of these cards
            probabilityOfWin1v1 = 1 - (len(possibleOppList) / float(52 - handSize))
            # I know this is wrong, but not very wrong, needs some sampling without replacement stuff...
            # Probability that no opposition players have one of these cards
            probabilityTotalVictory = probabilityOfWin1v1 ** (nplayers - 1)
            outputProbs.append(probabilityTotalVictory)
        return outputProbs

    def generateModelPlayer(self):
        modelPlayer = RandomPlayer(self.playerNumber, self.possibleHand)
        modelPlayer.canBidZero = self.canBidZero
        modelPlayer.zeroCounter = self.zeroCounter
        modelPlayer.cardsLeftInHand = self.cardsLeftInHand
        modelPlayer.monteCarloNumber = self.monteCarloNumber
        modelPlayer.realHand = [i for i in self.realHand]
        return copy.deepcopy(modelPlayer)