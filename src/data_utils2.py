import mne
import config
from utils import getAllFilesWithPaths
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pdb

class PreprocessDataAndEvents:
    def __init__(self, filepath, preload=False) -> None:
        self.filePath = filepath
        self.preload = preload
        print(f'Processing {self.filePath} file')
        self.preprocessDataFile()

    def preprocessDataFile(self):
        self.loadFifFile()
        self.getEventsDetails()
        self.groupEvents()

    def loadFifFile(self):
        self.rawData = mne.io.read_raw_fif(self.filePath, preload=self.preload, verbose=False)
        self.sampleFreq = self.rawData.info['sfreq']
        self.events = self.rawData.annotations
        

    def getEventsDetails(self):
        eventDetails = []
        for event in self.events:
            onset = event['onset']
            category, duration = self.mapCategory(event['description'])
            endTime = onset + duration
            startIndex = int(onset * self.sampleFreq)
            baselineStartIndex = startIndex - config.baselineWindow
            endIndex = int(endTime * self.sampleFreq)
            eventDetails.append([onset, category, baselineStartIndex, startIndex, endIndex, duration, endIndex - startIndex])

        self.eventDetails = [item for item in eventDetails if item[5] > 1]
        

    def groupEvents(self):
        eventDetailsArray = np.array(self.eventDetails, dtype=object)
        onset = eventDetailsArray[:, 0]
        categories = eventDetailsArray[:, 1]
        baselineStartIndex = eventDetailsArray[:, 2]
        startIndex = eventDetailsArray[:, 3]
        endIndex = eventDetailsArray[:, 4]
        duration = eventDetailsArray[:, 5]
        nSamples = eventDetailsArray[:, 6]

        splitCategories = [category.split('_') for category in categories]
        activity, modality, semantics = zip(*splitCategories)
        
        eventsDict = {
            'onset': onset, 'activity': activity, 'modality': modality,
            'semantics': semantics, 'baselineStartIndex': baselineStartIndex,
            'startIndex': startIndex, 'endIndex': endIndex,  
            'duration': duration, 'nSamples': nSamples
        }

        self.eventsCategorized = pd.DataFrame(eventsDict)
        
    def filterEvents(self, activity=None, modality=None, semantics=None):
        filtered = self.eventsCategorized
        if activity is not None:
            filtered = filtered[filtered['activity'] == activity]
        if modality is not None:
            filtered = filtered[filtered['modality'] == modality]
        if semantics is not None:
            filtered = filtered[filtered['semantics'] == semantics]

        self.filteredEvents = filtered
        
    def mapCategory(self,category):
        taskType = 'Perception'
        activityType = 'Audio'
        classType = 'Flower'
        if 'Perception' in category:
            if '_a_' in category or 'a_' in category:
                duration = 2
            elif '_image_' in category:
                duration = 3
                activityType = 'Image'
            elif '_t_' in category:
                duration = 3
                activityType = 'Text'
            else:
                activityType = ''
                duration = 0
        else:
            taskType = 'Imagination'
            if '_a_' in category or 'a_' in category:
                duration = 4
            elif '_image_' in category:
                duration = 4
                activityType = 'Image'
            elif '_t_' in category:
                duration = 4
                activityType = 'Text'
            else:
                activityType = ''
                duration = 0

        if 'flower' in category:
            pass
        elif 'guitar' in category:
            classType = 'Guitar'
        elif 'penguin' in category:
            classType = 'Penguin'
        else:
            classType = ''

        return f'{taskType}_{activityType}_{classType}', duration

class ImaginationPerceptionData:
    def __init__(self, bidsPath=config.processedDatasetDir, activity='Imagination', modality=None, semantics=None) -> None:
        self.activity = activity
        self.modality = modality
        self.semantics = semantics
        self.bidsPath = bidsPath
        self.plotCount = 0
        self.trialsData = []
        self.erpData = []
        self.filePaths = getAllFilesWithPaths(self.bidsPath)
        self.filePaths = [path for path in self.filePaths if 'eeg-1' not in path and 'sub-013' not in path]
        self.extractDataFromFiles()

    def extractDataFromFiles(self):
        self.subjectIds = []
        self.sessionIds = []
        for filePath in self.filePaths:
            subjectID, sessionID = self.extractIDs(filePath)
            self.subjectIds.append(subjectID)
            self.sessionIds.append(sessionID)
            print(f'Extraction ::: Subject {subjectID}: session {sessionID}')
            subject = PreprocessDataAndEvents(filepath=filePath, preload=True)
            subject.filterEvents(activity=self.activity, modality=self.modality, semantics=self.semantics)
            
            self.extractFilteredData(subject)

        self.saveCategoryDataOfTrials()
        
        #self.calculateERPForAllSubjects()
        #self.plotERPForAllSubjects()

    def saveCategoryDataOfTrials(self):
        print('Saving data')
        folderName = f'{self.activity}_{self.modality}_{self.semantics}'
        destinationFolder = Path(config.dataFolder, folderName)
        os.makedirs(destinationFolder, exist_ok=True)
        for index in range(len(self.trialsData)):
            data = self.trialsData[index]
            filename = f'{self.subjectIds[index]}_{self.sessionIds[index]}.npy'
            pathWithFilename = Path(destinationFolder, filename)
            print(pathWithFilename)
            np.save(pathWithFilename, data)

    def extractIDs(self, filePath):
        parts = filePath.split(config.seperator)
        return parts[-4], parts[-3]

    def extractFilteredData(self, subject):
        data = subject.rawData.pick(config.imaginationChannels if self.activity == 'Imagination' else config.perceptionChannels).get_data()
        individualTrials = []
        
        for _, event in subject.filteredEvents.iterrows():
            baselineStartIndex = int(event['baselineStartIndex'])
            startIndex = int(event['startIndex'])
            endIndex = startIndex + config.eventWindow * 1024
            trialData = data[:, baselineStartIndex:endIndex]
            individualTrials.append(trialData)
        individualTrials = np.array(individualTrials)
        
        self.trialsData.append(np.array(individualTrials))
        
    def calculateERPForAllSubjects(self):
        for trials in self.trialsData:
            self.erpData.append(np.mean(trials, axis=0))

    def plotERPForAllSubjects(self):
        for index, data in enumerate(self.erpData):
            self.plotERPForSubject(data, index)

    def plotERPForSubject(self, data, subjectID):
        pdb.set_trace()
        nChannels = data.shape[0]
        plotsPerRow = 8
        nRows = int(np.ceil(nChannels / plotsPerRow))
        
        fig, axes = plt.subplots(nRows, plotsPerRow, figsize=(50, 2 * nRows))
        axes = axes.flatten()
        for index in range(nChannels):
            ax = axes[index]
            ax.plot(data[index])
            ax.legend([config.imaginationChannels[index]], loc='upper right')
            ax.axvline(x=config.baselineWindow, color='r', linestyle='--', label='Stimulus Finish')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('')

        for j in range(nChannels, len(axes)):
            fig.delaxes(axes[j])
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        filepath = Path(os.getcwd(), 'Images', self.subjectIDs[subjectID], self.sessionIDs[subjectID])
        os.makedirs(filepath, exist_ok=True)
        fig.suptitle(f'{self.subjectIDs[subjectID]}_{self.sessionIDs[subjectID]}')
        name = f'{self.subjectIDs[subjectID]}_{self.sessionIDs[subjectID]}_{self.activity}_{self.modality}_{self.semantics}_ERP_1.png'
        fullFilePath = filepath / name
        print(filepath)
        print(name)
        fig.savefig(fullFilePath, dpi=200)

