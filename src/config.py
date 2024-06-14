import os
from pathlib import Path

seperator = '\\'
baselineWindow = 200

# directory paths
currentDir = os.getcwd()
numpyDataDir = Path(currentDir, 'NumpyData')
preprocessedDatasetDir = Path(currentDir, 'preprocessed')

# extraction 
startIndex = 200
endIndex = startIndex + 1024
batchSize = 32
epochs = 50


# transformer model architecture
numLayers = 4
embedDim = 128
numHeads = 8
dff = 512
inputVocabSize = 10000
maximumPositionEncoding = 10000
numClasses = 3

trainDataDir =  Path(numpyDataDir, 'SematicData')