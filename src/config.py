import os
from pathlib import Path

seperator = '\\'
baselineWindow = 200
segmentDataBasedOnSemantics = False
train = False
eval = True
trainModelName = 'initial_cnn.pth'
trainedModelName = 'tranformer.pth'
# directory paths
currentDir = os.getcwd()
numpyDataDir = Path(currentDir, 'NumpyData')
preprocessedDatasetDir = Path(currentDir, 'preprocessed')

trainDataDir =  Path(numpyDataDir, 'SematicData')
trainedModelDir = Path(currentDir, 'TrainedModels')
scaledDataDir = Path(numpyDataDir, 'ScaledSemantic')
# extraction 
startIndex = 200
endIndex = startIndex + 1024
batchSize = 256
epochs = 15


# transformer model architecture
inputDim = 1024 
seqLength = 124
dModel = 512 
nHead = 8
numEncoderLayers = 6 
dimFeedforward = 2048
dropout = 0.1
numClasses = 3

