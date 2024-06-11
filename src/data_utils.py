import mne
import config
from utils import getAllFilesWithPaths
import pdb

class PreprocessData:
    def __init__(self, filepath, preload=False) -> None:
        self.filePath = filepath
        self.rawData = None
        self.events = None
        self.sampleFreq = None
        self.preload = preload
        self.preprocessDataFile()

        

    def preprocessDataFile(self):
        self.loadFifFile()
        self.getEventsDetails()
        
    def loadFifFile(self):
        self.rawData = mne.io.read_raw_fif(self.filePath, preload=self.preload, verbose=False)
        self.sampleFreq = self.rawData.info['sfreq']

    def getEventsDetails(self):
        eventDetails = []
        self.events = self.rawData.annotations
        for event in self.events:
            onset = event['onset']
            category, duration = self.mapCategory(event['description'])
            startTime = onset - 1
            endTime = onset + duration
            startIndex = int(startTime * self.sampleFreq)
            endIndex = int(endTime * self.sampleFreq)
            eventDetails.append([onset, category, startIndex, endIndex, duration, endIndex-startIndex])
        
        self.eventDetails = [item for item in eventDetails if item[4] !=0]
        self.imaginationEvents = [item for item in self.eventDetails if 'Imagination' in item[1]]
        self.perceptionEvents = [item for item in self.eventDetails if 'Perception' in item[1]]
        

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
    def __init__(self, bidsPath = config.processedDatasetDir) -> None:
        self.bidsPath = bidsPath
        self.filePaths = getAllFilesWithPaths(self.bidsPath)
        self.filePaths = [path for path in self.filePaths if 'eeg-1' not in path]

        self.perceptionData = []
        self.ImaginationData = []

        self.extractDataFromFiles()


    def extractDataFromFiles(self):
        for filePath in self.filePaths:
            self.fileEventsDetails = PreprocessData(filepath=filePath)
            break
        
