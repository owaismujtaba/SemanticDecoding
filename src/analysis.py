import config
import os
from pathlib import Path
import numpy as np
import pdb
import mne
from matplotlib import pyplot as plt
from PIL import Image
import os
from utils import getAllPreprocessedFiles

def loadSementiciWiseData(activity=None, SemanticCategory=None, modality=None):
    dataDir = config.sementicDataDir
    classes = os.listdir(dataDir)
    data = []
    if SemanticCategory == None:
        for className in classes:
            classFolder = Path(dataDir, className)
            for filename in os.listdir(classFolder):
                if activity in filename and modality in filename:
                    data.appned(np.load(filename))

def cleanEvents(events):
    cleanedEvents = []
    ignoreEvents = [100, 200, 201]
    for event in events:
        if event[2] in ignoreEvents:
            continue
        else:
            cleanedEvents.append(event)
    cleanedEvents = np.array(cleanedEvents)
    return cleanedEvents

def perceptionAndImaginationEvents(events): 
    cleanedEvents = cleanEvents(events[0])
    perceptionKeys = []
    imaginationKeys = []
    for item, key in events[1].items():
        if 'Perception' in item:
            perceptionKeys.append(key)
        if 'Imagination' in item:
            imaginationKeys.append(key)

    for index in range(cleanedEvents.shape[0]):
        key = cleanedEvents[index][2]
        if key in perceptionKeys:
            cleanedEvents[index][2] = 1
        elif key in imaginationKeys:
            cleanedEvents[index][2] = 2

    return cleanedEvents

def erpAnalysisSingleSubject(filepath):
    print(f'Plotting for file {filepath}')
    subjectId = filepath.split(config.seperator)[-4]
    sessionId = filepath.split(config.seperator)[-3]
    data = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    events = mne.events_from_annotations(data)
    
    cleanedEvents = perceptionAndImaginationEvents(events)
    newEventId = {'Perception':1, 'Imagination':2}
    epochs = mne.Epochs(data.copy(), cleanedEvents, event_id=newEventId, tmin=-0.2, tmax=1.0)

    perception = epochs['Perception'].average()
    imagination = epochs['Imagination'].average()
    destinationDir = Path(config.ImagesDir, 'ERP')
    os.makedirs(destinationDir, exist_ok=True)
    perceptionFilename = f'{subjectId}_{sessionId}_Perception.png'
    perceptionFilenameWithPath = Path(destinationDir, perceptionFilename)
    imaginationFilename = f'{subjectId}_{sessionId}_Imagination.png'
    imaginationFilenameWithPath = Path(destinationDir, imaginationFilename)
    figureName = f'{subjectId}_{sessionId}_Perception-Imagination_ERP.png'
    figureNameWithPath = Path(destinationDir, figureName)

    
    
    perceptionFigure = perception.plot_joint(show=False)
    imaginationFigure = imagination.plot_joint(show=False)
    perceptionFigure.suptitle(f'Perception')
    imaginationFigure.suptitle(f'Imagination')

    perceptionFigure.savefig(perceptionFilenameWithPath, dpi = 600)
    imaginationFigure.savefig(imaginationFilenameWithPath, dpi = 600)
    
    image1 = Image.open(perceptionFilenameWithPath)
    image2 = Image.open(imaginationFilenameWithPath)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    figTitle = f'{subjectId} {sessionId}'
    ax1.imshow(image1)
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    #fig.suptitle(figTitle, fontsize=12)
    fig.savefig(figureNameWithPath, dpi=600,bbox_inches='tight', pad_inches=0)
    os.remove(perceptionFilenameWithPath)
    os.remove(imaginationFilenameWithPath)
    plt.close(fig)
   
def plotErpForAllSubjects():
    filepaths = getAllPreprocessedFiles()
    for filepath in filepaths:
        erpAnalysisSingleSubject(filepath)
        

plotErpForAllSubjects()