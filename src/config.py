import os
from pathlib import Path

device = 'CPU'
currentDir = os.getcwd()
resultsDir = Path(currentDir, 'Results')
trainedModelsDir = Path(resultsDir, "Models")
preprocessedDatasetDir = Path(currentDir, 'preprocessed')
dataDir = Path(currentDir, 'Data')
os.makedirs(resultsDir, exist_ok=True)

preprocessSemanticData = False
preprocessPerceptionImaginationData = False
preprocessPerceptionSemanticData = True

trainModelsOnPerceptionImaginationDecoding = False
trainModelsOnSemanticData = False

evaluatePerceptionImaginationDecoding = False

seperator = '/'
tmin = -0.2
tmax = 1.0
samplingFrequency = 1000

numEpochs = 100
batchSize = 256
numJobs = 5
numChannels = 124

speechThoughtIndexes = [
    4,  # F3
    5,  # F4
    3,  # F7
    7,  # F8
    6,  # Fz
    9,  # FC1
    10, # FC2
    38, # FC3
    39, # FC4
    30, # AF3
    31, # AF4
    28, # AF7
    29, # AF8
    66, # FT7
    67, # FT8
    13, # T7
    14, # T8
    55, # TP7
    56, # TP8
    15, # C3
    16, # C4
    17  # Cz
]

