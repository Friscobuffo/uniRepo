from utils.dataManager import Manager
from utils.computer import Computer
import pandas as pd

manager = Manager()
manager.autoGetData()

computer = Computer()
processedCoordinates = computer.compute(manager)

dataframe = pd.DataFrame(processedCoordinates, columns=["timestamps", "x", "y"])
dataframe.to_csv(manager.getOutputPath()+'/gazedata.csv', index=False)

print("Process ended without failure, output is located in /output folder")