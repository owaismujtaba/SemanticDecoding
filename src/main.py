from data_utils2 import ImaginationPerceptionData
import config
import mne
import numpy as np
from utils import PlotERP


#bidsPath = '/media/owais/UBUNTU 20_0/perceptionImaginationEEG/perceptionImaginationEEG/DataSet/derivatives/preprocessed'
bidsPath = 'C:\ImaginationPerceptionDataset\DataSet\perceptionImaginationEEG\preprocessed'
activities = ['Imagination', 'Perception']
modalities = [None, 'Audio', 'Text', 'Image']
sematices = [None, 'Guitar', 'Flower', 'Penguin']
for activity in activities:
    for modality in modalities:
        for category in sematices:
            print('Task')
            print(activity, modality, category)
            data = ImaginationPerceptionData(bidsPath=bidsPath, activity=activity, modality=modality, semantics=category)
            
    
       

PlotERP(config.dataFolder)
    
    