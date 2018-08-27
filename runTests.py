
from WhistStructure import *
from WhistLib import *
import subprocess

def parseTestConfig(testConfigFileName):
    testsToRun =[]
    with open(testConfigFileName) as data_file:
        jsonFileData = json.load(data_file)
        for test in jsonFileData["TestList"]:
            testName = test["TestName"]
            outputFileName = test["OutputFileName"]
            testsToRun.append([testName, outputFileName])
    return testsToRun

def runTests(testsToRun):
    for test in testsToRun:
        print("Starting Test...")
        with open(test[1], "w") as outputFileObject:
            subprocess.call(["python", "runGame.py", test[0]], stdout=outputFileObject)

if __name__ == "__main__":
    testConfigFileName = "./testConfigFiles/TestConfigFile.json"
    runTests( parseTestConfig(testConfigFileName) )