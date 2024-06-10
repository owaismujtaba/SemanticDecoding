import config
import os
from pathlib import Path
import mne
import pdb

def getDirFromFolder(folderPath = config.datasetDir):
    """
    Returns a list of directories within the specified folder.

    :param folderPath: Path to the folder
    :return: List of directories
    """
    try:
        # List all entries in the specified folder
        entries = os.listdir(folderPath)
        directories = [entry for entry in entries if os.path.isdir(os.path.join(folderPath, entry))]
        directories = [Path(folderPath, folder) for folder in directories]
        
        return directories
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

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


loadPatientData()