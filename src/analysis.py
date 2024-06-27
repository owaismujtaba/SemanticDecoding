import config
import os
from pathlib import Path
import numpy as np
import pdb
import mne
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from mne import Epochs
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
    epochs = mne.Epochs(data.copy(), cleanedEvents, event_id=newEventId, tmin=-0.2, tmax=1.0, preload=True)

    perception = epochs['Perception'].average()
    imagination = epochs['Imagination'].average()
    difference = mne.combine_evoked([perception, imagination], weights=[1, -1])
    destinationDir = Path(config.ImagesDir, 'ERP')
    os.makedirs(destinationDir, exist_ok=True)
    perceptionFilename = f'{subjectId}_{sessionId}_Perception.png'
    perceptionFilenameWithPath = Path(destinationDir, perceptionFilename)
    imaginationFilename = f'{subjectId}_{sessionId}_Imagination.png'
    imaginationFilenameWithPath = Path(destinationDir, imaginationFilename)
    figureName = f'{subjectId}_{sessionId}_Perception-Imagination_ERP.png'
    figureNameWithPath = Path(destinationDir, figureName)
    differenecFigureName = f'{subjectId}_{sessionId}_Perception-Imagination_ERP_Difference.png'
    differenecFigureNameWithPath = Path(destinationDir, differenecFigureName)
    
    perceptionFigure = perception.plot_joint(show=False)
    imaginationFigure = imagination.plot_joint(show=False)
    differenceFigure = difference.plot_joint(show=False)
    perceptionFigure.suptitle(f'Perception')
    imaginationFigure.suptitle(f'Imagination')
    differenceFigure.suptitle(f'Difference')

    perceptionFigure.savefig(perceptionFilenameWithPath, dpi = 600)
    imaginationFigure.savefig(imaginationFilenameWithPath, dpi = 600)
    differenceFigure.savefig(differenecFigureNameWithPath, dpi=600)
    
    image1 = Image.open(perceptionFilenameWithPath)
    image2 = Image.open(imaginationFilenameWithPath)

    del perceptionFigure
    del imaginationFigure
    del differenceFigure

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    ax1.imshow(image1)
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    fig.savefig(figureNameWithPath, dpi=600,bbox_inches='tight', pad_inches=0)
    os.remove(perceptionFilenameWithPath)
    os.remove(imaginationFilenameWithPath)
    plt.close(fig)

    clustersFileName = f'{subjectId}_{sessionId}_Perception-Imagination_ERP_Clusters.png'
    clustersFileNameWithPath = Path(destinationDir, clustersFileName)
    plotClustersERP(epochs, subjectId, sessionId, clustersFileNameWithPath)
   
def plotClustersERP(epochs, subjectId, sessionId, fileNameWithPath):
    evoked_perception = epochs['Perception'].average()
    data_evoked_perception = evoked_perception.data  

    Z_perception = linkage(data_evoked_perception, method='ward')
    num_clusters = 3

    cluster_labels_perception = fcluster(Z_perception, num_clusters, criterion='maxclust')
    clusters_perception = {}
    for i, label in enumerate(cluster_labels_perception):
        clusters_perception.setdefault(label, []).append(epochs.ch_names[i])

    evoked_imagination = epochs['Imagination'].average()
    data_evoked_imagination = evoked_imagination.data  

    Z_imagination = linkage(data_evoked_imagination, method='ward')

    cluster_labels_imagination = fcluster(Z_imagination, num_clusters, criterion='maxclust')
    clusters_imagination = {}
    for i, label in enumerate(cluster_labels_imagination):
        clusters_imagination.setdefault(label, []).append(epochs.ch_names[i])

    fig, axes = plt.subplots(2, num_clusters, figsize=(30, 12), sharex=True, sharey=True)

    for cluster_idx, (cluster, channels) in enumerate(clusters_perception.items()):
        evoked_cluster = evoked_perception.copy().pick(channels)

        ax = axes[0, cluster_idx]
        evoked_cluster.plot(axes=ax, show=False, time_unit='s')  

        ax.set_title(f"Perception Cluster {cluster}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(x=0, color='gray', linestyle='--')

    for cluster_idx, (cluster, channels) in enumerate(clusters_perception.items()):
        evoked_cluster = evoked_imagination.copy().pick(channels)

        ax = axes[1, cluster_idx]
        evoked_cluster.plot(axes=ax, show=False, time_unit='s')  

        ax.set_title(f"Imagination Cluster {cluster}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(x=0, color='gray', linestyle='--')

    fig.suptitle(f'{subjectId} {sessionId}', fontsize=16)
    fig.savefig(fileNameWithPath, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plotErpForAllSubjects1():
    filepaths = getAllPreprocessedFiles()
    for filepath in filepaths:
        if '017' in filepath or '018'  in filepath or '019'  in filepath:
            erpAnalysisSingleSubject(filepath)
        
def plotErpForAllSubjects():
    import concurrent.futures
    filepaths = getAllPreprocessedFiles()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(erpAnalysisSingleSubject, filepaths)

plotErpForAllSubjects1()