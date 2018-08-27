
from alphawhist.core.WhistStructure import *
from os import system

if __name__ == "__main__":
    scoreLog = []
    for i in range(0,1):
        system('title ' + 'game_number_'+str(i))
        thisGame = Game("./testConfigFiles/ThreePlayerRandom.json")
        thisGame.playFullGame()
        scoreLog.append( thisGame.scores )
    # f = open('manual.txt', 'w')
    # print(scoreLog, file=f)