
import json
from pprint import pprint

def doJson(configFileName):
    with open(configFileName) as data_file:
        jsonFileData = json.load(data_file)
        for parameterCollection in jsonFileData["GameParameters"]:
            playerConfigName = parameterCollection["PlayerConfigFile"]
            print playerConfigName

        #pprint(jsonFileData)

doJson("GameConfigFile.json")