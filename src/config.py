import os
from pathlib import Path

seperator = '\\'
baselineWindow = 200
segmentDataBasedOnSemantics = False
train = True
eval = False
analysis = False
trainedModelName = 'lstmModel.pth'
# directory paths
currentDir = os.getcwd()
numpyDataDir = Path(currentDir, 'NumpyData')
preprocessedDatasetDir = Path(currentDir, 'preprocessed')

trainDataDir =  Path(numpyDataDir, 'SematicData')
trainedModelDir = Path(currentDir, 'TrainedModels')
scaledSementicDataDir = Path(numpyDataDir, 'ScaledSemantic')
sementicDataDir = Path(numpyDataDir, 'SematicData')
ImagesDir = Path(currentDir, 'Images')

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
numClasses = 2



channelNames = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
    'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 
    'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 
    'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 
    'P6', 'PO3', 'PO4', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h',
    'TPP10h', 'PO9', 'PO10', 'P9', 'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 
    'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 
    'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', 
    'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h', 
    'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 
    'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 
    'OI2h']
perception_channels = [
    'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO7', 'PO8', 'PO9', 'PO10',
    'POO9h', 'POO10h', 'POO3h', 'POO4h', 'OI1h', 'OI2h', 'P7', 'P3', 
    'Pz', 'P4', 'P8', 'CP5', 'CP1', 'CP2', 'CP6', 'P5', 'P1', 'P2', 
    'P6', 'PPO1', 'PPO2', 'PPO9h', 'PPO10h', 'PPO5h', 'PPO6h'
]

imagination_channels = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'AF7', 'AF3', 
    'AF4', 'AF8', 'AFF1', 'AFz', 'AFF2', 'F5', 'F1', 'F2', 'F6', 
    'AFF5h', 'AFF6h', 'CPP1h', 'CPP2h', 'CPP3h', 'CPP4h', 'CPP5h', 
    'CPP6h', 'CCP1h', 'CCP2h', 'CCP3h', 'CCP4h', 'CCP5h', 'CCP6h'
]
