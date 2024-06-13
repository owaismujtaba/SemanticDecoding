from data_utils1 import ImaginationPerceptionData
import config
import mne
import numpy as np


bidsPath = '/media/owais/UBUNTU 20_0/perceptionImaginationEEG/perceptionImaginationEEG/DataSet/derivatives/preprocessed'
#bidsPath = 'G:\perceptionImaginationEEG\perceptionImaginationEEG\DataSet\derivatives\preprocessed'
activities = ['Imagination', 'Perception']
modalities = [None, 'Audio', 'Text', 'Image']
sematices = [None, 'Guitar', 'Flower', 'Penguin']
for activity in activities:
    for modality in modalities:
        for category in sematices:
            print('Task')
            print(activity, modality, category)
            data = ImaginationPerceptionData(bidsPath=bidsPath, activity=activity, modality=modality, semantics=category)