
from WhistStructure import *
from WhistLib import *

if __name__ == "__main__":
    scoreLog = []
    for i in range(0,20):
        thisGame = Game("./testConfigFiles/ThreePlayerMonteVsAdvancedBid1000.json")
        thisGame.playFullGame()
        scoreLog.append( thisGame.scores )
    f = open('logfile.txt', 'w')
    print >> f, scoreLog
    