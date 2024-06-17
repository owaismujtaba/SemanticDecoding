import mne
import numpy as np
import pandas as pd
import os
from pathlib import Path
import threading
from sklearn.preprocessing import MinMaxScaler

from utils import getAllPreprocessedFiles
import config
import pdb

class PreprocessDataAndEvents:
    def __init__(self, filepath, preload=False, segmentData=False, scaling=False):
        self.filePath = filepath
        self.preload = preload
        self.segementDataFlag = segmentData
        self.scaling = scaling
        filepath = filepath.split(config.seperator)
        self.subjectId = filepath[-4]
        self.sessionId = filepath[-3]
        print(f'Processing {self.subjectId} {self.sessionId} file')
        
        self.preprocessDataFile()

    def preprocessDataFile(self):
        self.loadFifFile()
        self.getEventsDetails()
        self.groupEvents()
        self.splitEventsBySemantics()
        if self.segementDataFlag:
            self.extractDataBySemantics()

    def loadFifFile(self):
        self.rawData = mne.io.read_raw_fif(self.filePath, preload=self.preload, verbose=False)
        self.sampleFreq = self.rawData.info['sfreq']
        self.events = self.rawData.annotations

    def getEventsDetails(self):
        eventDetails = []
        for event in self.events:
            onset = event['onset']
            category, duration = self.mapCategory(event['description'])
            startIndex = int(onset*self.sampleFreq)
            baselineStartIndex = startIndex - config.baselineWindow
            endIndex = startIndex + int(duration*self.sampleFreq)
            eventDetails.append([onset, category, baselineStartIndex, startIndex, endIndex, duration, endIndex - startIndex])
            

        self.eventDetails = [item for item in eventDetails if item[5] > 1]
        
    def groupEvents(self):
        eventDetailsArray = np.array(self.eventDetails, dtype=object)
        columns = ['onset', 'category', 'baselineStartIndex', 'startIndex', 'endIndex', 'duration', 'nSamples']
        eventsDict = {col: eventDetailsArray[:, idx] for idx, col in enumerate(columns)}

        splitCategories = [category.split('_') for category in eventsDict['category']]
        activity, modality, semantics = zip(*splitCategories)
        
        eventsDict.update({
            'activity': activity,
            'modality': modality,
            'semantics': semantics
        })

        self.eventsCategorized = pd.DataFrame(eventsDict)
        self.eventsCategorized[['activityType', 'modalityType', 'semanticType']] = self.eventsCategorized['category'].str.split('_', expand=True)
        self.eventsCategorized.drop('category', inplace=True, axis=1)
        del self.eventDetails

    def mapCategory(self, category):
        taskType = 'Perception' if 'Perception' in category else 'Imagination'
        if '_a_' in category or 'a_' in category:
            activityType = 'Audio'
            duration = 2 if taskType == 'Perception' else 4
        elif '_image_' in category:
            activityType = 'Image'
            duration = 3 if taskType == 'Perception' else 4
        elif '_t_' in category:
            activityType = 'Text'
            duration = 3 if taskType == 'Perception' else 4
        else:
            activityType = ''
            duration = 0

        if 'flower' in category:
            classType = 'Flower'
        elif 'guitar' in category:
            classType = 'Guitar'
        elif 'penguin' in category:
            classType = 'Penguin'
        else:
            classType = ''

        return f'{taskType}_{activityType}_{classType}', duration
    
    def splitEventsBySemantics(self):
        semanticTypes = self.eventsCategorized['semanticType'].unique()
        self.semanticTypesEvents = {semanticType: self.eventsCategorized[self.eventsCategorized['semanticType'] == semanticType].reset_index(drop=True) for semanticType in semanticTypes}

    def extractDataBySemantics(self):
        numpyRawData = self.rawData.copy().get_data()
        if self.scaling:
            print(f'Normalizing data')
            dirDestination = Path(config.scaledDataDir)
            scaler = MinMaxScaler()
            numpyRawData = scaler.fit_transform(numpyRawData)
        else:
            dirDestination = Path(config.numpyDataDir, 'SematicData')
        
        for semnticCategory, events in self.semanticTypesEvents.items():
            dirCategory = Path(dirDestination, semnticCategory)
            os.makedirs(dirCategory, exist_ok=True)
            baselineStartIndexes = events['baselineStartIndex']
            endIndexs = events['endIndex']
            for index in range(events.shape[0]):
                activity = events['activity'][index]
                modality = events['modality'][index]
                filename = f'{self.subjectId}_{self.sessionId}_{activity}_{modality}_{semnticCategory}_{index}'
                filenameWithPath = Path(dirCategory, filename)
                startIndex = baselineStartIndexes[index]
                endIndex = endIndexs[index]
                data = numpyRawData[:, startIndex:endIndex]
                np.save(filenameWithPath, data)
                print(f'Saved {filenameWithPath}')

class SemanticSegmentation:
    def __init__(self, scaling=False) -> None:
        self.filepaths = getAllPreprocessedFiles()
        self.threading = False
        self.scaling = scaling
        

    def segmentFiles(self):
        threads = []
        for filepath in self.filepaths:
            if self.threading:
                thread = threading.Thread(target=self.preprocessFile, args=(filepath,))
                threads.append(thread)
                thread.start()
                for thread in threads:
                    thread.join()
                    self.preprocessFile(filepath)
            else:
                self.preprocessFile(filepath)

    def preprocessFile(self, filepath):
        #pdb.set_trace()
        PreprocessDataAndEvents(filepath, segmentData=True, scaling=self.scaling)