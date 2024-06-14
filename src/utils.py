import config
import os
from pathlib import Path
import pdb
import numpy as np
from matplotlib import pyplot as plt

class PlotERP:
    def __init__(self, datasetDir) -> None:
        self.datsetDir = datasetDir
        self.destionationDir = config.plotsDir
        self.getAllFilePaths()
        
    def getAllFilePaths(self):
        asllFilePaths = []
        _, categoryWithPaths = getDirFromFolder(self.datsetDir)
        for categoryDir in categoryWithPaths:
            category = str(categoryDir).split(config.seperator)[-1]
            filesWithPaths = getNumpyFilePaths(categoryDir)
            for filePath in filesWithPaths:
                filename = filename = filePath.split(config.seperator)[-1].split('.')[0]
                subject, session = filename.split('_')
                data = np.load(filePath)
                self.plotERPAllChannels(data,category, subject, session )
                
    def plotERPAllChannels(self, data,category, subject, session):
        
        print(f'Plotting for {subject} {session} {category}')
        directory = Path(self.destionationDir, subject)
        directory = Path(directory, session)
        os.makedirs(directory, exist_ok=True)
        filename = f'{category}_{subject}_{session}_ERP.png'
        filenameWithPath = Path(directory, filename)
        erpData = np.mean(data, axis=0)

        nChannels = data.shape[1]
        plotsPerRow = 8
        nRows = int(np.ceil(nChannels / plotsPerRow))
        
        fig, axes = plt.subplots(nRows, plotsPerRow, figsize=(50, 2 * nRows))
        axes = axes.flatten()
        for index in range(nChannels):
            ax = axes[index]
            ax.plot(erpData[index], color='grey')
            ax.plot(erpData[index][:config.baselineWindow], color='black')
            if 'Imagination' in category:
                ax.legend([config.imaginationChannels[index]], loc='upper right')
            if 'Perception' in category:
                ax.legend([config.perceptionChannels[index]], loc='upper right')
            ax.axvline(x=config.baselineWindow, color='r', linestyle='--', label='Stimulus Finish')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('')

        for j in range(nChannels, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(f'{category}_{subject}_{session}')
        fig.savefig(filenameWithPath, dpi=600)
        

class SemanticDataImagination:
    def __init__(self, dirPath) -> None:
        self.dirPath = dirPath
        self.guitarData = None
        self.flowerData = None
        self.penguinData = None
        self.loadSemanticDataFromAllSubjects()
        self.data = np.vstack((self.guitarData, self.flowerData))
        self.data = np.vstack((self.data, self.penguinData))
        self.guitarLabels = np.array([0]*self.guitarData.shape[0]).reshape(-1, 1)
        self.flowerLabels = np.array([1]*self.flowerData.shape[0]).reshape(-1, 1)
        self.penguinLables = np.array([2]*self.penguinData.shape[0]).reshape(-1, 1)
        self.labels = np.vstack((self.guitarLabels, self.flowerLabels))
        self.labels = np.vstack((self.labels, self.penguinLables))
        del self.guitarData
        del self.flowerData
        del self.penguinData

        

    def loadSemanticDataFromAllSubjects(self):
        _, categoryWithPaths = getDirFromFolder(self.dirPath)
        for categoryDir in categoryWithPaths:
            categoryNameParts = str(categoryDir).split(os.sep)[-1].split('_')
            semanticCategory = categoryNameParts[-1]
            activityCategory = categoryNameParts[0]
            modalityCategory = categoryNameParts[1]

            if semanticCategory!='None' and activityCategory=='Imagination' and modalityCategory=='None':
                filesWithPaths = getNumpyFilePaths(categoryDir)
                for filePath in filesWithPaths:
                    print(semanticCategory, categoryDir)
                    data = np.load(filePath)
                    self.appendData(semanticCategory, data)
        
    def appendData(self, semanticCategory, data):
        if semanticCategory == 'Guitar':
            self.guitarData = self.stackData(self.guitarData, data)
        elif semanticCategory == 'Penguin':
            self.penguinData = self.stackData(self.penguinData, data)
        elif semanticCategory == 'Flower':
            self.flowerData = self.stackData(self.flowerData, data)

    @staticmethod
    def stackData(existingData, newData):
        if existingData is not None:
            return np.vstack((existingData, newData))
        else:
            return newData


#####################fif Files Related########################
def getNumpyFilePaths(directory):
    """
    Returns a list of full paths of all .npy files in the given directory.

    Parameters:
    directory (str): The path to the directory to search for .npy files.

    Returns:
    List[str]: A list containing the full paths of .npy files.
    """
    numpyFiles = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                numpyFiles.append(os.path.join(root, file))
    return numpyFiles

def listFifFiles(directory):
    fifFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.fif'):
                filePath = os.path.join(root, file)
                fifFiles.append(filePath)
    
    return fifFiles

def getDirFromFolder(folderPath = config.datasetDir):
    """
    Returns a list of directories within the specified folder.

    :param folderPath: Path to the folder
    :return: List of directories
    """
    try:
        entries = os.listdir(folderPath)
        directories = [entry for entry in entries if os.path.isdir(os.path.join(folderPath, entry))]
        directoriesWithPaths = [Path(folderPath, folder) for folder in directories]
        return directories, directoriesWithPaths
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []
    
def getAllFilesWithPaths(fifDir = config.rawDatasetDir):
    allFilesWithPaths = []
    
    _, subjectDirNamesWithPaths = getDirFromFolder(fifDir)
    for subjectDirPath in subjectDirNamesWithPaths:
        _, sessionDirNameWithPaths = getDirFromFolder(subjectDirPath)
        for sessionDirPath in sessionDirNameWithPaths:
            dirPath = Path(sessionDirPath, 'eeg')
            files = listFifFiles(dirPath)
            for file in files:
                allFilesWithPaths.append(file)
   
        
    return allFilesWithPaths



#####################FDT Files Related########################

