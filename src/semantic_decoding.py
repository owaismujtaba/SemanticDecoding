import os
import warnings
import numpy as np
from pathlib import Path

from src.models import EEGNetBasedModel, EEGNet, EegNetSsvepN, DeepConvNet
from src.models import SVCModel, RandomForestModel, XGBoostModel
from src.trainner import EEGModelTrainer
from src.utils import loadExtractedFeatures, loadCSPFeatures
import src.config as config




warnings.filterwarnings('ignore')

import pdb

class SemanticData:
    def __init__(self, activity=None, cleaned=False):
        self.name = f'{activity}Semantic'
        self.dataDir = Path(config.dataDir, self.name)
        if cleaned:
            print(f'***Loading {self.name} cleaned data***')
            self.dataDir = Path(self.dataDir, 'Cleaned') 
        else:
            print(f'***Loading {self.name}processed data***')
            self.dataDir = Path(self.dataDir, 'Processed')
        self.destinationDir = config.resultsDir
        
        self.loadSemanticData()

    def loadSemanticData(self):
        files = os.listdir(self.dataDir)
        print(self.dataDir)
        print('Loading Semantic Data for All Subjects')
        for file in files:
            filepath = Path(self.dataDir, file)
            if 'Session' in file:
                self.sessionIds = np.load(filepath)
            elif 'Subject' in file:
                self.subjectIds = np.load(filepath)
            elif 'xTrain' in file:
                self.xTrain = np.load(filepath)
            elif 'xTest' in file:
                self.xTest = np.load(filepath)
            elif 'yTrain' in file:
                self.yTrain = np.load(filepath)
            elif 'yTest' in file:
                self.yTest = np.load(filepath)
            elif 'TestSizes' in file:
                self.testSizes = np.load(filepath)
        print('Loaded Semantic for All Subjects')

def trainAllModelsForSemanticDecoding():
    taskType = 'Semantic'
    numClasses =  3
    dataLoader = SemanticData(cleaned=True)
    xTrain, xTest = dataLoader.xTrain, dataLoader.xTest
    yTrain, yTest = dataLoader.yTrain, dataLoader.yTest
    
    #dataDir = Path(config.dataDir, taskType, 'CSPFeatures')
    #xTrain, xTest = loadCSPFeatures(dataDir)
   
    xTrainFeatures, xTestFeatures = loadExtractedFeatures(
            folder=Path(config.dataDir, taskType)
    )
    #xTrainFeatures = xTrainFeatures.reshape(xTrainFeatures.shape[0], -1)
    #xTestFeatures = xTestFeatures.reshape(xTrainFeatures.shape[0], -1)
   
    xTrain = np.concatenate((xTrain, xTrainFeatures), axis=2)
    xTest =  np.concatenate((xTest, xTestFeatures), axis=2)

    
    xTrain = xTrain[:,config.speechThoughtIndexes,:]
    xTest = xTest[:, config.speechThoughtIndexes,:]

    
    print(f'Xtrain: {xTrain.shape}, xTest: {xTest.shape}')
    
    del dataLoader
    '''
    #Machine Learning Based Models Training
    trainner = SVCModel(
        numClasses=numClasses,
        taskType=taskType        
    )
    trainner.hyperParameterTunning(
        xTrain=xTrain,
        xTest=xTest,
        yTrain = yTrain,
        yTest=yTest
    )
    trainner = RandomForestModel(
        numClasses=numClasses,
        taskType=taskType        
    )
    trainner.hyperParameterTunning(
        xTrain=xTrain,
        xTest=xTest,
        yTrain = yTrain,
        yTest=yTest
    )
     








    #Deep Learning Based Models Training
    '''
    
    modelBuilder = EEGNetBasedModel(numClasses=numClasses,chans=22, samples=1059)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName='SpEEGNetBasedModel',
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    '''
    modelBuilder = EEGNet(numClasses=numClasses, samples=22)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="SpEEGNet",
        taskType=taskType
    )
    
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    
    modelBuilder = EegNetSsvepN(numClasses=numClasses, samples=22)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="SpEEGNetSsvepN",
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    
    modelBuilder = DeepConvNet(numClasses=numClasses, samples=22)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="SpDeepConvNet",
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    '''
    
