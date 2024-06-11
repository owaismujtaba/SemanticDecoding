import config
import os
from pathlib import Path
import pdb




#####################fif Files Related########################
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

