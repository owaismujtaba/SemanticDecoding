import mne
import config
from utils import getAllFilesWithPaths
import pdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PreprocessDataAndEvents:
    def __init__(self, filepath, preload=False) -> None:
        self.filePath = filepath
        self.imaginationRawData = None
        self.perceptionRawData = None
        self.auditoryRawData = None
        self.rawData = None
        self.events = None
        self.sampleFreq = None
        self.eventsCategorized = None
        self.preload = preload
        print(f'processing {self.filePath} file')
        self.preprocessDataFile()

    def preprocessDataFile(self):
        self.loadFifFile()
        self.getEventsDetails()
        self.groupEvents()
        
    def loadFifFile(self):
        self.rawData = mne.io.read_raw_fif(self.filePath, preload=self.preload, verbose=False)
        self.sampleFreq = self.rawData.info['sfreq']
        self.imaginationRawData = self.rawData.copy().pick(config.imaginationChannels)
        self.perceptionRawData = self.rawData.copy().pick(config.perceptionChannels)
        self.auditoryRawData = self.rawData.copy().pick(config.auditoryChannels)

    def getEventsDetails(self):
        eventDetails = []
        self.events = self.rawData.annotations
        for event in self.events:
            onset = event['onset']
            category, duration = self.mapCategory(event['description'])
            endTime = onset + duration
            startIndex = int(onset * self.sampleFreq)
            baselineStartIndex = startIndex - config.baselinewWindow
            endIndex = int(endTime * self.sampleFreq)
            eventDetails.append([onset, category, baselineStartIndex, startIndex, endIndex, duration, endIndex-startIndex])

        self.eventDetails = [item for item in eventDetails if item[5] >1]

    def groupEvents(self):
        onset = []
        activity = []
        modality = []
        semantics = []
        baselineStartIndex = []
        startIndex = []
        endIndex = []
        duration = []
        nSamples = []

        for event in self.eventDetails:
            onset_, category_, baselineStartIndex_, startIndex_, endIndex_, duration_, nSamples_ = event
            activity_, modality_, semantics_ = category_.split('_')
            onset.append(onset_)
            activity.append(activity_)
            modality.append(modality_)
            semantics.append(semantics_)
            baselineStartIndex.append(baselineStartIndex_)
            startIndex.append(startIndex_)
            endIndex.append(endIndex_)
            duration.append(duration_)
            nSamples.append(nSamples_)

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
    def __init__(self, 
                bidsPath = config.processedDatasetDir,
                activity = 'Imagination',
                modality = None,
                semantics = None
        ) -> None:
        self.activity = activity
        self.modality = modality
        self.semantics = semantics
        self.bidsPath = bidsPath
        self.trialsData = []
        self.erpData = []
        self.filePaths = getAllFilesWithPaths(self.bidsPath)
        self.filePaths = [path for path in self.filePaths if 'eeg-1' not in path]
        self.subjectData = []
        self.extractDataFromFiles()

    def extractDataFromFiles(self):
        for filePath in self.filePaths[:1]:
            subject = PreprocessDataAndEvents(filepath=filePath, preload=True)
            self.subjectData.append(subject)
            subject.filterEvents(activity=self.activity, 
                                        modality=self.modality, 
                                        semantics=self.semantics     
                                    )
            self.extractFilteredData(subject, self.activity)
            self.calculateERPForAllSubjects()

    def extractFilteredData(self, subject, activity):
        if activity == 'Imagination':
            data = subject.imaginationRawData.get_data()
        else:
            data = subject.perceptionRawData.get_data()
        individualTrials = []
        for trialIndex in range(subject.filteredEvents.shape[0]):
            baselineStartIndex = subject.filteredEvents.iloc[trialIndex]['baselineStartIndex']
            startIndex = subject.filteredEvents.iloc[trialIndex]['startIndex']
            endIndex = startIndex + config.eventWindow * 1024
            trialData = data[:, baselineStartIndex:endIndex]
            individualTrials.append(trialData)
        self.trialsData.append(np.array(individualTrials))
        
    def calculateERPForAllSubjects(self):
        for index in range(len(self.trialsData)):
            self.calculateERPForSubject(self.trialsData[index])

    def calculateERPForSubject(self, data):
        erpData = np.mean(data, axis=0)
        self.erpData.append(erpData)


    def plotERPForAllSubjects(self):
        for index in range(len(self.erpData)):
            self.plotERPForSubject(self, self.erpData[index], index)

    def plotERPForSubject(data, subjectID):
        nChannels = data.shape[0]
        n_samples = data.shape[1]
        plotsPerRow = 8
        nRows = int(np.ceil(nChannels/plotsPerRow))

        fig, axes = plt.subplots(nRows, plotsPerRow, figsize=(50,2*nRows))
        axes = axes.flatten()
        for index in range(nChannels):
            ax = axes[index]
            ax.plot(data[index])
            ax.legend([config.imaginationChannels[i]], loc='upper right')  
            ax.axvline(x=config.baselinewWindow, color='r', linestyle='--', label='Stimulus Finish')
            ax.set_ylabel('Amplitude (uV)')
            ax.spines['top'].set_visible(False)  
            ax.spines['right'].set_visible(False)  
            ax.set_xlabel('')  

        for j in range(nChannels, len(axes)):
            fig.delaxes(axes[j])

        fig.savefig(f'{subjectID}.png', dpi=600)