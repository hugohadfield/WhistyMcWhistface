from alphawhist.core.WhistStructure import *

if __name__ == "__main__":
	playerConfigName = sys.argv[1]
	thisGame = Game(playerConfigName)
	thisGame.playFullGame()