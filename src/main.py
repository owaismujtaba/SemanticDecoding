from data_utils import ImaginationPerceptionData
import config
import mne
import numpy as np


#bidsPath = 'F:\perceptionImaginationEEG\perceptionImaginationEEG\DataSet\derivatives\preprocessed'
bidsPath = 'G:\perceptionImaginationEEG\perceptionImaginationEEG\DataSet\derivatives\preprocessed'
data = ImaginationPerceptionData(bidsPath=bidsPath)