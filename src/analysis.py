import config
import os
from pathlib import Path
import numpy as np
import pdb
import mne

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


def erpAnalysis(filepath):
    data = mne.io.read_raw_fif(filepath, preload=True, verbose=False)

    events = mne.events_from_annotations(data)
    eventId = {f'Imagination': list(range(1, 101)), f'Perception': list(range(101, 200))}

    mappedEvents = []
    for event in events[0]:
        if event[2] in eventId['Imagination']:
            mappedEvents.append([event[0], event[1], 1])
        elif event[2] in eventId['Perception']:
            mappedEvents.append([event[0], event[1], 2])
    mappedEvents = np.array(mappedEvents)

    newEventId = {'Perception':2, 'Imagination':1}
    epochs = mne.Epochs(data.copy(), mappedEvents[2:], event_id=newEventId, tmin=-0.2, tmax=1.0)

    perception = epochs['Perception'].average()
    imagination = epochs['Imagination'].average()

    perFig = perception.plot()
    imagFig = imagination.plot()

    perFig.savefig('Perception.png', dpi=600)
    imagFig.savefig('imagination.png', dpi=600)