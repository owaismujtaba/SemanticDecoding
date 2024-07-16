from pathlib import Path
import os
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from tensorflow.keras.models import load_model
from  src import config
import pdb
import mne
from scipy.stats import zscore
from mne.decoding import CSP
import pywt
import joblib
import multiprocessing as mp

import src.config as config






def computeSegmentFeatures(segment, sfreq, scales, freqBands):
    nChannels = segment.shape[0]
    
    means = np.mean(segment, axis=1)
    stds = np.std(segment, axis=1)
    skewnesses = skew(segment, axis=1)
    kurts = kurtosis(segment, axis=1)
    
    freqs, psds = welch(segment, sfreq, axis=1, nperseg=2*sfreq)
    
    bandPowers = [np.sum(psds[:, (freqs >= low) & (freqs < high)], axis=1) for low, high in freqBands.values()]
    bandPowers = np.array(bandPowers).T
    
    morletFeatures = [np.sum(np.abs(pywt.cwt(segment[ch], scales, 'cmor1.5-1.0', sampling_period=1/sfreq)[0])**2, axis=1) for ch in range(nChannels)]
    morletFeatures = np.array(morletFeatures)
    
    segmentFeatures = np.hstack((
        means[:, np.newaxis],
        stds[:, np.newaxis],
        skewnesses[:, np.newaxis],
        kurts[:, np.newaxis],
        bandPowers,
        morletFeatures
    ))

    return segmentFeatures

def extractFeatures(
        segmentedData, 
        name='xTrain',
        sfreq=config.samplingFrequency
    ):
    print('*********Extracting Features************')
    
    nSegments, nChannels, nTimes = segmentedData.shape
    
    freqBands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100),
        
    }
    
    frequencies = np.linspace(1, sfreq / 2, 50)
    scales = pywt.scale2frequency('cmor1.5-1.0', frequencies) * sfreq
    
    with mp.Pool(processes=config.numJobs) as pool:
        results = pool.starmap(
            computeSegmentFeatures, 
            [(segmentedData[i], sfreq, scales, freqBands) for i in range(nSegments)]
        )
    
    features = np.array(results)
    
    destinationDir = Path(config.dataDir, 'Features')
    os.makedirs(destinationDir, exist_ok=True)
    filenameWithPath = Path(destinationDir, f'{name}.npy')
    np.save(filenameWithPath, features)
    return features


def loadExtractedFeatures(folder):
    print('***********Loading Features*************')
    folder = Path(folder, 'Features')
    files = os.listdir(folder)
    
    for file in files:
        if 'Train' in file:
            xTrainFeatures = np.load(Path(folder, file))
        elif 'Test' in file:
            xTestFeatures = np.load(Path(folder, file))
    
    return xTrainFeatures, xTestFeatures


def cleanData(mneData):
    print("*************claening the data**************")
    data = mneData.copy()
    data.filter(l_freq=0.5, h_freq=150)
    data.set_eeg_reference("average", projection=True)
    ica = mne.preprocessing.ICA(
        n_components=20, 
        random_state=97,
        max_iter=800
    )

    ica.fit(data)
    
    ica.apply(data)
    dataStandardized = zscore(data.get_data(), axis=1)
    
    data._data = dataStandardized
    return data

def loadCSPFeatures(folder):
    print('***********Loading CSP Features*************')
    files = os.listdir(folder)
    
    for file in files:
        if 'Train' in file:
            xTrainFeatures = np.load(Path(folder, file))
        elif 'Test' in file:
            xTestFeatures = np.load(Path(folder, file))
    
    return xTrainFeatures, xTestFeatures

def getCSPFeatures(data, labels):
    print('Extracting CSP Features')
    csp = CSP(
        n_components=20,
        reg=None,
        log=None,
        cov_est="concat"
    )
    csp.fit(data, labels)
    transformedData = csp.transform(data)

    return csp, transformedData


def getLDAFeatures(data, labels):
    print('Extracting LDA Features')
    data = data.reshape(data.shape[0], -1)
    lda = LinearDiscriminantAnalysis()
    transformdData = lda.fit_transform(data, labels)
    return lda, transformdData

def saveTensorFlowModel(model, modelNameWithPath, modelType="Tensor"):
    if modelType != "Tensor":
        pickle.dump(model, open(modelNameWithPath, "wb"))
    else:
        model.save(modelNameWithPath)



 
def getAllFifFilesFromFolder(directory):
    fifFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.fif') and 'eeg-1' not in file:
                filePath = os.path.join(root, file)
                fifFiles.append(filePath)
    return fifFiles

def getDirFromFolder(folderPath):
    entries = os.listdir(folderPath)
    directories = [entry for entry in entries if os.path.isdir(os.path.join(folderPath, entry))]
    directoriesWithPaths = [Path(folderPath, folder) for folder in directories]
    return directories, directoriesWithPaths

def getAllPreprocessedFiles(folderPath=config.preprocessedDatasetDir):
    allFilePaths = []
    _, subjectFolders = getDirFromFolder(folderPath)
    for subject in subjectFolders:
        _, sessionFolders = getDirFromFolder(subject)
        for session in sessionFolders:
            dirPath = Path(session, 'eeg')
            files = getAllFifFilesFromFolder(dirPath)
            for file in files:
                allFilePaths.append(file)
                
    allFilePaths = [filepath for filepath in allFilePaths if 'sub-013' not in filepath]
    
    
    return allFilePaths