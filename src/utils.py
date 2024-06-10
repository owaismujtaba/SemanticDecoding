import config
import os
from pathlib import Path
import mne
from mne.io import read_raw_eeglab
from pyprep.prep_pipeline import PrepPipeline
from pyprep import find_noisy_channels

import pdb



def getFilesFromFolder(folderPath):
    """
    Recursively get all file paths from the given directory.
    
    Args:
    directory (str): The directory from which to retrieve file paths.
    
    Returns:
    list: A list of file paths.
    """
    filePaths = []
    
    for root, directories, files in os.walk(folderPath):
        for filename in files:
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)
    
    return filePaths


def loadPatientData(patiendID='sub-10'):

    folders = getDirFromFolder()

    for folder in folders:
        print(folder)
        filesInCurrentFolder = getFilesFromFolder(folder)

        for file in filesInCurrentFolder:
            pdb.set_trace()
            data = mne.io.read_raw_fif(file, preload=True)



#####################FDT Files Related########################
def listFdtFiles(directory):
    fdtFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(file)
            if not file.endswith('.fdt'):
                filePath = os.path.join(root, file)
                fdtFiles.append(filePath)
    
    return fdtFiles

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
        return []
    
def getALlFdtFilesWithPaths(fdtDir = config.rawDatasetDir):
    fdtFilesPathsWithSubjectId = []
    fdtSubjextDirsNames, fdtSubjectDirsWithPaths = getDirFromFolder(fdtDir)
    for dirName, dirPath in zip(fdtSubjextDirsNames, fdtSubjectDirsWithPaths):
        filesInDir = listFdtFiles(dirPath)
        for file in filesInDir:
            fdtFilesPathsWithSubjectId.append([dirName, file])

    return fdtFilesPathsWithSubjectId

def loadRawFdtFile(filePath, eog=config.eogChannels):
    pdb.set_trace()
    rawData = read_raw_eeglab(filePath, preload=True)
    print(rawData.info)
    return rawData

#####################FDT Files Related########################


def preprocessRawFdtFiles():
    filesWithSubjetIds = getALlFdtFilesWithPaths(config.rawDatasetDir)
    
    for subjectID, filePath in filesWithSubjetIds:
        print(f'Loading {subjectID}')
        rawData = loadRawFdtFile(filePath)
        

