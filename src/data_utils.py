import mne
from autoreject import get_rejection_threshold
import pdb

class PreprocessData:
    def __init__(self, filepath) -> None:
        self.filePath = filepath
        self.rawData = None
        self.epochs = None
        self.events = None
        self.sampleFreq = None
        self.preprocessDataFile()

    def preprocessDataFile(self):
        self.loadFifFile()
        pdb.set_trace()
        self.getEventsDetails()
        

    def loadFifFile(self):
        self.rawData = mne.io.read_raw_fif(self.filePath)
        self.sampleFreq = self.rawData.info['sfreq']

    def getEventsDetails(self):
        eventDetails = []
        events = self.rawData.annotations
        for event in events:
            onset = event['onset']
            category, duration = self.mapCategory(event['description'])
            startTime = onset - 1
            endTime = onset + duration
            startIndex = int(startTime * self.sampleFreq)
            endIndex = int(endTime * self.sampleFreq)
            eventDetails.append([onset, category, startIndex, endIndex, duration])
        self.eventsDetails = eventDetails

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
        

filePath =  '/media/owais/UBUNTU 20_0/perceptionImaginationEEG/perceptionImaginationEEG/DataSet/derivatives/preprocessed/sub-03/ses-03/eeg/sub3_sess3_50_ica_eeg-1.fif'
data = PreprocessData(filePath)