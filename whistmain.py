
from WhistStructure import *
from WhistLib import *

def gametest(strategyList):
    thisgame = Game(strategyList)
    for i in range(0,len(thisgame.cardnumbers)):
        print("Round number: ", i+1)

        thisgame.playFullRound()
    return thisgame

if __name__ == "__main__":
    #dakestest()
    strategyList = ["advanced", "randomUncontrolled", "randomUncontrolled"]
    scoreLog = []
    for i in range(0,5):
        finishedGame = gametest(strategyList)
        scoreLog.append( finishedGame.scores )

    f = open('logfile', 'w')
    print >> f, scoreLog
    print '\n'