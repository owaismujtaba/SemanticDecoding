import numpy as np
import mne
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


import src.config as config
from src.utils import cleanData, extractFeatures
from src.utils import getAllPreprocessedFiles, getCSPFeatures, getLDAFeatures
from src.perception_imagination_decoding import PerceptionImaginationData
from src.semantic_decoding import SemanticData

import pdb



class PerceptionSemanticDataProcessor:
    def __init__(self):
        self.name = 'PerceptionSemantic'
        self.destinationDir = Path(config.dataDir, self.name)
        #self.destinationDir = Path(self.destinationDir, 'Processed')
        

    def getPerceptionSemanticDataOfSubject(self, filepath):
        print('***************** Loading Perception Semantic Data *********************')
        print(f'File: {filepath}')
        data = mne.io.read_raw_fif(filepath, verbose=False, preload=True)
        data = cleanData(data)
        events, eventIds = mne.events_from_annotations(data, verbose=False)
        eventIdsReversed = {str(value): key for key, value in eventIds.items()}
        
        codes, eventTimings = [], []
        for event in events:
            eventCode = eventIdsReversed.get(str(event[2]), None)
            if eventCode:
                code = self._getCode(eventCode)
                if code:
                    codes.append(code)
                    eventTimings.append(event[0])
        semanticEvents = [[timing, 0, code] for timing, code in zip(eventTimings, codes)]
        semanticEventIds = {'flower': 1, 'guitar': 2, 'penguin': 3}
        
        epochs = mne.Epochs(data.copy(), semanticEvents, event_id=semanticEventIds, tmin=config.tmin, tmax=config.tmax, preload=True, verbose=False)
        flowerData, guitarData, penguinData = epochs['flower'].get_data(), epochs['guitar'].get_data(), epochs['penguin'].get_data()

        flowerData, guitarData, penguinData = flowerData[:, :, 230:], guitarData[:, :, 230:], penguinData[:, :, 230:]
        labels = np.concatenate(([0] * len(flowerData), [1] * len(guitarData), [2] * len(penguinData)), axis=0)
        
        data = np.concatenate((flowerData, guitarData, penguinData), axis=0)
        print(f'X: {data.shape}, Y {labels.shape}')
        print('***************** Loaded Perception Semantic Data *********************')
        return data, labels

    @staticmethod
    def _getCode(event):
        if 'Perception' in event:
            if 'flower' in event:
                return 1
            elif 'guitar' in event:
                return 2
            elif 'penguin' in event:
                return 3
            return None   
        else:
            return None 
        
    def preprocessPerceptionSementicDataAllSubjects(self):
        print('****Perception Semantic Data Preprocessing****')
        self.destinationDir = Path(self.destinationDir, "Cleaned")
        os.makedirs(self.destinationDir, exist_ok=True)
        
        filepaths = getAllPreprocessedFiles()
        subjectIds = []
        sessionIds = []
        trainData = None
        trainLabels = None
        testData = None
        testLabels = None
        testSizes = []
        
        for index in range(len(filepaths)):
            subjectIds.append(filepaths[index].split(config.seperator)[-4])
            sessionIds.append(filepaths[index].split(config.seperator)[-3])
            X, y = self.getPerceptionSemanticDataOfSubject(filepaths[index])
            xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if index == 0:
                trainData = xTrain
                trainLabels = yTrain
                testData = xTest
                testLabels = yTest
                testSizes.append(xTest.shape[0])   
            else:
                trainData =  np.concatenate((trainData, xTrain), axis=0)
                trainLabels = np.concatenate((trainLabels, yTrain))
                testData =  np.concatenate((testData, xTest), axis=0)
                testLabels = np.concatenate((testLabels, yTest), axis=0)
                testSizes.append(xTest.shape[0])
            
        
        
        print(f'Saving files to {self.destinationDir} directory')
        np.save(Path(self.destinationDir, 'xTrain.npy'), trainData)
        np.save(Path(self.destinationDir, 'yTrain.npy'), trainLabels)
        np.save(Path(self.destinationDir, 'xTest.npy'), testData)
        np.save(Path(self.destinationDir, 'yTest.npy'), testLabels)
        np.save(Path(self.destinationDir, 'SubjectIds.npy'), subjectIds)
        np.save(Path(self.destinationDir, 'SessionIds.npy'), sessionIds)
        np.save(Path(self.destinationDir, 'TestSizes.npy'), testSizes)

def perceptionSemanticPreProcessingPipiline():
    name = 'PerceptionSemantic'
    
    #perceptionSemanticDataProcessor = PerceptionSemanticDataProcessor()
    #perceptionSemanticDataProcessor.preprocessPerceptionSementicDataAllSubjects()
    perceptionSemnaticDataLoader = SemanticData(activity = 'Perception', cleaned=True)
    xTrain, xTest = perceptionSemnaticDataLoader.xTrain, perceptionSemnaticDataLoader.xTest
    yTrain, yTest = perceptionSemnaticDataLoader.yTrain, perceptionSemnaticDataLoader.yTest
    
    
    # Extracting Statistical Features, morlet features and PSD features
    xTrainFeatures = extractFeatures(xTrain)
    xTestFeatures = extractFeatures(xTest)
    pdb.set_trace()
    destinationDir = Path(config.dataDir, name)
    destinationDir = Path(destinationDir, 'Features')
    os.makedirs(destinationDir, exist_ok=True)

    np.save(Path(destinationDir, 'xTrain.npy'), xTrainFeatures)
    np.save(Path(destinationDir, 'xTest.npy'), xTestFeatures)

    print('Cleand and Features Extracted for Semantic Data')
    
    destinationDir = Path(config.dataDir, name)
    destinationDir = Path(destinationDir, 'CSPFeatures')
    os.makedirs(destinationDir, exist_ok=True)
    cspModel, xTrainFeatures = getCSPFeatures(xTrain, yTrain)
    xTestFeatures = cspModel.transform(xTest)
    np.save(Path(destinationDir, 'xTrain.npy'), xTrainFeatures)
    np.save(Path(destinationDir, 'xTest.npy'), xTestFeatures)

class SemanticDataProcessor:
    def __init__(self):
        self.name = 'Semantic'
        self.destinationDir = Path(config.dataDir, self.name)
        self.destinationDir = Path(self.destinationDir, 'Processed')
        

    def getSemanticDataOfSubject(self, filepath):
        print('***************** Loading Semantic Data *********************')
        print(f'File: {filepath}')
        data = mne.io.read_raw_fif(filepath, verbose=False, preload=True)
        data = cleanData(data)
        events, eventIds = mne.events_from_annotations(data, verbose=False)
        eventIdsReversed = {str(value): key for key, value in eventIds.items()}
        
        codes, eventTimings = [], []
        for event in events:
            eventCode = eventIdsReversed.get(str(event[2]), None)
            if eventCode:
                code = self._getCode(eventCode)
                if code:
                    codes.append(code)
                    eventTimings.append(event[0])
        semanticEvents = [[timing, 0, code] for timing, code in zip(eventTimings, codes)]
        semanticEventIds = {'flower': 1, 'guitar': 2, 'penguin': 3}
        
        epochs = mne.Epochs(data.copy(), semanticEvents, event_id=semanticEventIds, tmin=config.tmin, tmax=config.tmax, preload=True, verbose=False)
        flowerData, guitarData, penguinData = epochs['flower'].get_data(), epochs['guitar'].get_data(), epochs['penguin'].get_data()

        flowerData, guitarData, penguinData = flowerData[:, :, 230:], guitarData[:, :, 230:], penguinData[:, :, 230:]
        labels = np.concatenate(([0] * len(flowerData), [1] * len(guitarData), [2] * len(penguinData)), axis=0)
        
        data = np.concatenate((flowerData, guitarData, penguinData), axis=0)
        print(f'X: {data.shape}, Y {labels.shape}')
        print('***************** Loaded Semantic Data *********************')
        return data, labels

    @staticmethod
    def _getCode(event):
        if 'flower' in event:
            return 1
        elif 'guitar' in event:
            return 2
        elif 'penguin' in event:
            return 3
        return None    

    def preprocessSementicDataAllSubjects(self):
        print('****Semantic Data Preprocessing****')
        self.destinationDir = Path(self.destinationDir, "Cleaned")
        os.makedirs(self.destinationDir, exist_ok=True)
        
        filepaths = getAllPreprocessedFiles()
        subjectIds = []
        sessionIds = []
        trainData = None
        trainLabels = None
        testData = None
        testLabels = None
        testSizes = []
        
        for index in range(len(filepaths)):
            subjectIds.append(filepaths[index].split(config.seperator)[-4])
            sessionIds.append(filepaths[index].split(config.seperator)[-3])
            X, y = self.getSemanticDataOfSubject(filepaths[index])
            xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if index == 0:
                trainData = xTrain
                trainLabels = yTrain
                testData = xTest
                testLabels = yTest
                testSizes.append(xTest.shape[0])   
            else:
                trainData =  np.concatenate((trainData, xTrain), axis=0)
                trainLabels = np.concatenate((trainLabels, yTrain))
                testData =  np.concatenate((testData, xTest), axis=0)
                testLabels = np.concatenate((testLabels, yTest), axis=0)
                testSizes.append(xTest.shape[0])
        
        print(f'Saving files to {self.destinationDir} directory')
        np.save(Path(self.destinationDir, 'xTrain.npy'), trainData)
        np.save(Path(self.destinationDir, 'yTrain.npy'), trainLabels)
        np.save(Path(self.destinationDir, 'xTest.npy'), testData)
        np.save(Path(self.destinationDir, 'yTest.npy'), testLabels)
        np.save(Path(self.destinationDir, 'SubjectIds.npy'), subjectIds)
        np.save(Path(self.destinationDir, 'SessionIds.npy'), sessionIds)
        np.save(Path(self.destinationDir, 'TestSizes.npy'), testSizes)


class PerceptionImaginationDataProcessor:
    def __init__(self):
        self.name = 'PerceptionImagination'
        self.destinationDir = Path(config.dataDir, self.name)
        

    def getPerceptionImaginationDataOfSubject(self, filepath):
        print('***************** Loading Perception/Imagination Data *********************')
        print(f'File: {filepath}')
        data = mne.io.read_raw_fif(filepath, verbose=False, preload=True)  
        data = cleanData(data)
        events, eventIds = mne.events_from_annotations(data, verbose=False)
        eventIdsReversed = {str(value): key for key, value in eventIds.items()}
        codes, eventTimings = [], []
        for event in events:
            eventCode = eventIdsReversed.get(str(event[2]), None)
            if eventCode:
                code = self._getCode(eventCode)
                if code:
                    codes.append(code)
                    eventTimings.append(event[0])

        perceptionImaginationEvents = [[timing, 0, code] for timing, code in zip(eventTimings, codes)]
        perceptionImaginationEventIds = {'Perception': 1, 'Imagination': 2}
        
        epochs = mne.Epochs(data.copy(), perceptionImaginationEvents, event_id=perceptionImaginationEventIds, tmin=config.tmin, tmax=config.tmax, preload=True, verbose=False)
        perceptionData, imaginationData = epochs['Perception'].get_data(), epochs['Imagination'].get_data()

        
        perceptionData, imaginationData = perceptionData[:, :, 230:], imaginationData[:, :, 230:]
        labels = np.concatenate(([0] * len(perceptionData), [1] * len(imaginationData)), axis=0)
        
        data = np.concatenate((perceptionData, imaginationData), axis=0)
        print(f'X: {data.shape}, Y {labels.shape}')
        print('***************** Loaded Perception/Imagination Data *********************')
        return data, labels

        

    @staticmethod
    def _getCode(event):
        if 'Perception' in event:
            return 1
        elif 'Imagination' in event:
            return 2
        else:
            return None



    def preprocessPerceptionImaginationDataAllSubjects(self):
        print('****Perception/Imagination Data Preprocessing****')
        self.destinationDir = Path(self.destinationDir, "Cleaned")
        os.makedirs(self.destinationDir, exist_ok=True)
        
        filepaths = getAllPreprocessedFiles()
        subjectIds = []
        sessionIds = []
        trainData = None
        trainLabels = None
        testData = None
        testLabels = None
        testSizes = []
        
        for index in range(len(filepaths)):
            subjectIds.append(filepaths[index].split(config.seperator)[-4])
            sessionIds.append(filepaths[index].split(config.seperator)[-3])
            X, y = self.getPerceptionImaginationDataOfSubject(filepaths[index])
            xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

            
            if index == 0:
                trainData = xTrain
                trainLabels = yTrain
                testData = xTest
                testLabels = yTest
                testSizes.append(xTest.shape[0])   
            else:
                trainData =  np.concatenate((trainData, xTrain), axis=0)
                trainLabels = np.concatenate((trainLabels, yTrain))
                testData =  np.concatenate((testData, xTest), axis=0)
                testLabels = np.concatenate((testLabels, yTest), axis=0)
                testSizes.append(xTest.shape[0])
            
    
        print(f'Saving files to {self.destinationDir} directory')
        np.save(Path(self.destinationDir, 'xTrain.npy'), trainData)
        np.save(Path(self.destinationDir, 'yTrain.npy'), trainLabels)
        np.save(Path(self.destinationDir, 'xTest.npy'), testData)
        np.save(Path(self.destinationDir, 'yTest.npy'), testLabels)
        np.save(Path(self.destinationDir, 'SubjectIds.npy'), subjectIds)
        np.save(Path(self.destinationDir, 'SessionIds.npy'), sessionIds)
        np.save(Path(self.destinationDir, 'TestSizes.npy'), testSizes)


def semanticPreProcessingPipiline():
    name = 'Semantic'
    
    semanticDataProcessor = SemanticDataProcessor()
    #semanticDataProcessor.preprocessSementicDataAllSubjects()
    semnaticDataLoader = SemanticData(cleaned=True)
    xTrain, xTest = semnaticDataLoader.xTrain, semnaticDataLoader.xTest
    yTrain, yTest = semnaticDataLoader.yTrain, semnaticDataLoader.yTest
    
    
    # Extracting Statistical Features, morlet features and PSD features
    xTrainFeatures = extractFeatures(xTrain)
    xTestFeatures = extractFeatures(xTest)
    
    destinationDir = Path(config.dataDir, name)
    destinationDir = Path(destinationDir, 'Features')
    os.makedirs(destinationDir, exist_ok=True)

    np.save(Path(destinationDir, 'xTrain.npy'), xTrainFeatures)
    np.save(Path(destinationDir, 'xTest.npy'), xTestFeatures)

    print('Cleand and Features Extracted for Semantic Data')
    
    destinationDir = Path(config.dataDir, name)
    destinationDir = Path(destinationDir, 'CSPFeatures')
    os.makedirs(destinationDir, exist_ok=True)
    cspModel, xTrainFeatures = getCSPFeatures(xTrain, yTrain)
    xTestFeatures = cspModel.transform(xTest)
    np.save(Path(destinationDir, 'xTrain.npy'), xTrainFeatures)
    np.save(Path(destinationDir, 'xTest.npy'), xTestFeatures)

def perceptionImaginationPreProcessingPipeline():
    name = 'PerceptionImagination'
    # Filtering, Averaging and Standardizing the data
    perceptionImaginationDataProcessor = PerceptionImaginationDataProcessor()
    perceptionImaginationDataProcessor.preprocessPerceptionImaginationDataAllSubjects()
    perceptionImaginationDataLoader = PerceptionImaginationData(cleaned=True)
    xTrain, xTest = perceptionImaginationDataLoader.xTrain, perceptionImaginationDataLoader.xTest
    yTrain, yTest = perceptionImaginationDataLoader.yTrain, perceptionImaginationDataLoader.yTest
    
    
    # Extracting Statistical Features, morlet features and PSD features
    xTrainFeatures = extractFeatures(xTrain)
    xTestFeatures = extractFeatures(xTest)

    destinationDir = Path(config.dataDir, name)
    destinationDir = Path(destinationDir, 'Features')
    os.makedirs(destinationDir, exist_ok=True)

    np.save(Path(destinationDir, 'xTrain.npy'), xTrainFeatures)
    np.save(Path(destinationDir, 'xTest.npy'), xTestFeatures)

    print('Cleand and Features Extracted for Perception and Imagination')
    
    #Extracting CSP Features
    destinationDir = Path(config.dataDir, name)
    destinationDir = Path(destinationDir, 'CSPFeatures')
    os.makedirs(destinationDir, exist_ok=True)
    cspModel, xTrainFeatures = getCSPFeatures(xTrain, yTrain)
    xTestFeatures = cspModel.transform(xTest)
    np.save(Path(destinationDir, 'xTrain.npy'), xTrainFeatures)
    np.save(Path(destinationDir, 'xTest.npy'), xTestFeatures)
    

   























