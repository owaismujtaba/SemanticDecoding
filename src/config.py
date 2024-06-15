import os
from pathlib import Path

seperator = '\\'
baselineWindow = 200
segmentDataBasedOnSemantics = False
train = True
inference = False
trainModelName = 'initial.pth'
trainedModelName = 'initial.pth'
# directory paths
currentDir = os.getcwd()
numpyDataDir = Path(currentDir, 'NumpyData')
preprocessedDatasetDir = Path(currentDir, 'preprocessed')

# extraction 
startIndex = 200
endIndex = startIndex + 1024
batchSize = 128
epochs = 10


# transformer model architecture
inputDim = 1024 
seqLength = 124
dModel = 512 
nHead = 8
numEncoderLayers = 6 
dimFeedforward = 2048
dropout = 0.1
numClasses = 3

trainDataDir =  Path(numpyDataDir, 'SematicData')
trainedModelDir = Path(currentDir, 'TrainedModels')