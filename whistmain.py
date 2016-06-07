
from WhistStructure import *
from WhistLib import *

def gametest():
    thisgame = Game()
    for i in range(0,len(thisgame.cardnumbers)):
        print("Round number: ", i+1)
        thisgame.playFullRound()
    return thisgame

if __name__ == "__main__":
    #dakestest()
    finishedGame = gametest()
    print finishedGame.scores
    