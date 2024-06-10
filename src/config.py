import os
from pathlib import Path

currentDir = os.getcwd()
datasetDir = Path(currentDir, 'Dataset')
rawDatasetDir = Path(currentDir, 'DataSet\derivatives\\fdt_files')

eogChannels = ['VEOGL', 'VEOGU', 'HEOGR', 'HEOGL']

auditoryChannels = ['T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'TP7', 'TP8', 'FT9', 'FT10', 'TTP7h', 'TTP8h']

imaginationChannels = ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
perceptionChannels = ['P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'PO3', 'PO4', 'PPO1', 'PPO2', 'PO9', 'PO10']
