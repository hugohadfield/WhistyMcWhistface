
from WhistStructure import *
from WhistLib import *
            
def dakestest():
     # Run some fuckin code
     for i in range(0,10):
        hand = generate_random_hand(7)
        print(hand)
        print(picktrump(hand))
            
def modeltest():
    thismodel = BotModel(generate_random_hand(7))
    print( all_players_beating_probability(thisgame.players[0].possiblehand[0], 'h', thisgame.players[1:3])    )

def gametest():
    thisgame = Game()
    for i in range(0,len(thisgame.cardnumbers)):
        print("Round number: ", i+1)
        thisgame.playRound()

if __name__ == "__main__":
    #dakestest()
    gametest()
        
        
        
    
    
    