import pyxdf
import mne
from datetime import datetime

import pdb


class XDFData:
    def __init__(self, filepath=None) -> None:
        self.rawData = None
        self.filepath = filepath
        self.data,  self.header = self.loadXdfData()
        self.setupData()
        self.printInfo()
        self.createMNEObjectForEEG()
        self.makeAnnotations()
        pdb.set_trace()
    def loadXdfData(self):
        print(f'Loading XDF File ::: {self.filepath}')
        self.data, self.header = pyxdf.load_xdf(self.filepath)
    
    def setupData(self,):
        self.eegChannelNames = []
        self.measDate = self.header['info']['datetime'][0]
        self.markers = self.data[1]['time_series']
        self.markers.pop()
        self.markers = [marker[0] for marker in self.markers]
        self.markersTimestamps = self.data[1]['time_stamps']
        self.eegData = self.data[2]['time_series']
        self.eegSamplingFrequency = int(float(self.data[2]['info']['nominal_srate'][0]))
        self.eegTimestamps = self.data[2]['time_stamps']
        self.audioData = self.data[3]['time_series']
        self.audioTimestamps = self.data[3]['time_stamps']
        self.audioSamplingFrequency = int(float(self.data[3]['info']['nominal_srate'][0]))
        channelNames = self.data[2]['info']['desc'][0]['channels'][0]['channel']
        for item in channelNames:
            self.eegChannelNames.append(item['label'][0])
    
    def printInfo(self):
        print(f'No of Markers: {len(self.markers)} No .of Marker Timestamps: {self.markersTimestamps.shape[0]}')
        print(f'EEG data Shape: {self.eegData.shape} No .of eeg Timestamps: {self.eegTimestamps.shape[0]}')
        print(f'Audio data Shape: {self.audioData.shape} No .of audio Timestamps: {self.audioTimestamps.shape[0]}')
        print('EEG Channels:', self.eegChannelNames)
        print(f'Sampling Frequency ::: EEG: {self.eegSamplingFrequency}, Audio: {self.audioSamplingFrequency}')

    def createMNEObjectForEEG(self):
        meas_date = datetime.strptime(self.measDate, '%Y-%m-%dT%H:%M:%S%z')
        meas_date = (int(meas_date.timestamp()), int(meas_date.microsecond))
        info = mne.create_info(
            ch_names=self.eegChannelNames, 
            sfreq=self.eegSamplingFrequency, 
            ch_types='eeg',
        )
        info.set_meas_date(meas_date)
        self.rawEegMNEData = mne.io.RawArray(self.eegData.T, info)
    def getOnsetCodesForAnnotations(self):
        codes = []
        onset = []
        description = []
        duration = []
        markers = self.markers
        markersTimestamps = self.markersTimestamps - self.eegTimestamps[0]
        for index in range(len(markers)-1):
            marker = markers[index]
            code = ''
            if 'Silent' in marker:
                code += '10' 
            if 'Real' in marker:
                code += '11' 
            if 'Word' in marker:
                code += '12' 
            if 'Syllable' in marker:
                code += '13' 
            if 'Practice' in marker:
                code += '14' 
            if 'Experiment' in marker:
                code += '15' 
            if 'Start' in marker:
                code += '16' 
            if 'End' in marker:
                code += '17' 
            if 'Fixation' in marker:
                code += '18' 
            elif 'Stimulus' in marker:
                code += '19' 
            elif 'ISI' in marker:
                code += '20' 
            elif 'ITI' in marker:
                code += '21' 
            elif 'Speech' in marker:
                code += '22' 
            else:
                print(marker)
            
            if 'Audio' in marker:
                code += '23'
            elif 'Text' in marker:
                code += '24'
            elif 'Speech' in marker:
                code += '25'

            codes.append(code)
            onset.append(markersTimestamps[index])
            description.append(marker)
            duration.append(markersTimestamps[index+1]-markersTimestamps[index])

        return onset, codes, duration

    def makeAnnotations(self):
        onset, codes, duration = self.getOnsetCodesForAnnotations()
        rawMNEWithAnnotations = self.rawEegMNEData.copy()
        rawMNEWithAnnotations.set_annotations(mne.Annotations(onset=onset, description=codes, duration=duration))

        self.rawMNEWithAnnotations = rawMNEWithAnnotations


filepath = 'C:\PilotData\sub-Pilooto1_ses-S001_task-Default_run-001_eeg.xdf'
obj = XDFData(filepath) 