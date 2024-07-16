import os
import numpy as np
import pandas as pd
from pathlib import Path
import pdb
import src.config as config
import warnings

from src.utils import loadExtractedFeatures, loadCSPFeatures
from src.models import EEGNet, EegNetSsvepN, DeepConvNet, EEGNetBasedModel
from src.models import XGBoostModel, SVCModel, RandomForestModel
from src.trainner import EEGModelTrainer
from src.eval import loadAllTrainedModels, classificationReport, getIndividualSpecificClassificationReport

warnings.filterwarnings('ignore')

class PerceptionImaginationData:
    def __init__(self, cleaned=False):
        self.name = 'PerceptionImagination'
        self.dataDir = Path(config.dataDir, self.name)
        if cleaned:
            print('***Loading cleaned data***')
            self.dataDir = Path(self.dataDir, 'Cleaned') 
        else:
            print('***Loading processed data***')
            self.dataDir = Path(self.dataDir, 'Processed')
        self.destinationDir = config.resultsDir
        
        self.loadImaginationPerceptionData()
        
    def loadImaginationPerceptionData(self):
        files = os.listdir(self.dataDir)
        print(self.dataDir)
        print('Loading Perception-Imagination Data for All Subjects')
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
        print('Loaded Perception-Imagination for All Subjects')




def trainAllModelsForPerceptionAndImaginationDecoding():
    taskType = 'PerceptionImagination'
    numClasses =  2
    dataLoader = PerceptionImaginationData(cleaned=True)
    xTrain, xTest = dataLoader.xTrain, dataLoader.xTest
    yTrain, yTest = dataLoader.yTrain, dataLoader.yTest
    
    
    xTrainFeatures, xTestFeatures = loadExtractedFeatures(
        folder=Path(config.dataDir, taskType)
    )
    xTrain = np.concatenate((xTrain, xTrainFeatures), axis=2)
    xTest =  np.concatenate((xTest, xTestFeatures), axis=2)
    print(f'Xtrain: {xTrain.shape}, xTest: {xTest.shape}')
    '''
    dataDir = Path(config.dataDir, taskType, 'CSPFeatures')
    xTrain, xTest = loadCSPFeatures(dataDir)
    
    
    del dataLoader
   
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
    
    modelBuilder = EEGNetBasedModel(numClasses=numClasses)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName='EEGNetBasedModel',
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    
    modelBuilder = EEGNet(numClasses=numClasses, samples=1059)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="EEGNet",
        taskType=taskType
    )
    
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    '''
    modelBuilder = EegNetSsvepN(numClasses=numClasses, samples=1059)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="EEGNetSsvepN",
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    
    modelBuilder = DeepConvNet(numClasses=numClasses, samples=1059)
    model = modelBuilder.buildModel()
    trainner = EEGModelTrainer(
        model=model,
        modelName="DeepConvNet",
        taskType=taskType
    )
    trainner.train(
        xTrain=xTrain,
        yTrain=yTrain,
        xVal=xTest,
        yVal=yTest
    )
    



def evaluateAllModelsForPerceptionAndImaginationDecoding():
    taskType = 'PerceptionImagination'
    destinationDir = Path(config.resultsDir, 'Reports')
    destinationDir = Path(destinationDir, taskType)
    os.makedirs(destinationDir, exist_ok=True)
    dataLoader = PerceptionImaginationData(cleaned=True)
    xTrain, xTest = dataLoader.xTrain, dataLoader.xTest
    yTrain, yTest = dataLoader.yTrain, dataLoader.yTest

    xTrainFeatures, xTestFeatures = loadExtractedFeatures(
        folder=Path(config.dataDir, taskType )
    )

    xTrain = np.concatenate((xTrain, xTrainFeatures), axis=2)
    xTest =  np.concatenate((xTest, xTestFeatures), axis=2)
    testSizes = dataLoader.testSizes
    sessionIds, subjectIds =  dataLoader.sessionIds, dataLoader.subjectIds
    
    loadedModels, modelNames = loadAllTrainedModels()

    for modelNo in range(0, len(loadedModels)):
        modelName = modelNames[modelNo]
        model = loadedModels[modelNo]
        
        if modelName == 'RF':
            continue
        machineLearningModels = ['XGB', 'SVC', 'RF']
        print(f'****************Model name {modelName}')
        if modelName in machineLearningModels:
            continue
            dataDir = Path(config.dataDir,taskType, 'CSPFeatures')
            _, xTest1 = loadCSPFeatures(dataDir)
            
            reportOnAllSubjects = classificationReport(model, xTest1, yTest)
            reportOnIndividualSubjects = getIndividualSpecificClassificationReport(
                model=model, xTest=xTest1, yTest=yTest,
                subjectIds=subjectIds, sessionIds=sessionIds,
                testSizes=testSizes    
            )
        else:
           
            print(xTest.shape, yTest.shape)
            reportOnAllSubjects = classificationReport(model, xTest, yTest)
            trainAccuracy = classificationReport(model, xTrain, yTrain)
            reportOnIndividualSubjects = getIndividualSpecificClassificationReport(
                model=model, xTest=xTest, yTest=yTest,
                subjectIds=subjectIds, sessionIds=sessionIds, 
                testSizes=testSizes   
            )
            print('Test Report')
            print(reportOnAllSubjects)
            print('Train report')
            print(trainAccuracy)
        
        reportOnAllSubjects = pd.DataFrame(reportOnAllSubjects)
        reportOnAllSubjects.to_csv(Path(destinationDir, f'{modelName}_AllSubjects.csv'))
        reportOnIndividualSubjects.to_csv(Path(destinationDir, f'{modelName}_IndividualSubjects.csv'))






















    
        
