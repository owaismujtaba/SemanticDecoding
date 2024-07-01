import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import getAllPreprocessedFiles
from src import config
import pdb



def cleanEvents(events):
    cleanedEvents = []
    ignoreEvents = [100, 200, 201]
    for event in events[0]:
        if event[2] in ignoreEvents:
            continue
        else:
            cleanedEvents.append(event)
    cleanedEvents = np.array(cleanedEvents)
    return cleanedEvents

def perceptionAndImaginationEvents(events): 
    cleanedEvents = cleanEvents(events)
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
    
def getPerceptionImaginationDataFromSubject(filepath):

    print(f'Loading MNE Data {filepath}')
    data = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    events = mne.events_from_annotations(data)
    cleanedEvents = perceptionAndImaginationEvents(events)
    newEventId = {'Perception':1, 'Imagination':2}
    epochs = mne.Epochs(data.copy(), cleanedEvents, event_id=newEventId, tmin=-0.2, tmax=1.0, preload=True)
    perception = epochs['Perception'].get_data()
    imagination = epochs['Imagination'].get_data()
    print(f'*******************************************')
    print(perception.shape, imagination.shape)
    return perception, imagination

def createXAndY(filepath, start=230):
    perception, imagination =  getPerceptionImaginationDataFromSubject(filepath)
    perception = perception[:,:,start:]
    imagination = imagination[:,:,start:]
    imagLables= np.array([0 for i in range(imagination.shape[0])])
    perceptionLabels = np.array([1 for i in range(perception.shape[0])])
    labels = np.concatenate((imagLables, perceptionLabels), axis=0)
    X = np.concatenate((imagination, perception), axis=0)
    X_flat = X.reshape(X.shape[0], -1)
    
    return X_flat, labels


def trainModel(filepath):
    X, y = createXAndY(filepath)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    print(X_train.shape, X_test.shape)
    print('Applying PCA for Dimentionality Reduction')
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('Training model')
    model = RandomForestClassifier(max_depth=30,
                                min_samples_split=10,
                                max_features='log2',
                                oob_score=True
                            )
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    report = classification_report(y_test, y_pred, target_names=['Imagination', 'perception'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm
   

def runRandomForestModelOnALlSubjects():
    filepaths = getAllPreprocessedFiles()
    subjectIds = []
    sessionIds = []
    average_precision = []
    avergare_recall = []
    average_f1_score = []
    #pdb.set_trace()
    for filepath in filepaths:
        subjectId = filepath.split('\\')[-4]
        sessionId = filepath.split('\\')[-3]
        subjectIds.append(subjectId)
        sessionIds.append(sessionId)
        report, cm = trainModel(filepath)

        average_precision.append(report['weighted avg']['precision'])
        avergare_recall.append(report['weighted avg']['recall'])
        average_f1_score.append(report['weighted avg']['f1-score'])

        name = f'{subjectId}_{sessionId}'
        picName = f'Confusion_Matrix_{name}.png'
        picNameWithPath = Path(config.ImagesDir, picName)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Imagination', 'perception'], yticklabels=['Imagination', 'perception'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(picNameWithPath, dpi=600)

    data = {'SubjectID': subjectIds,
        'SessionID':sessionIds,
        'Precision':average_precision,
        'Recall':avergare_recall,
        'f1-score': average_f1_score
        }
    data = pd.DataFrame(data)
    
    data.to_csv('Subjects_RF_metrics.csv')

def getAllPerceptionAndImaginationDataFromAllSubjects():
    filepaths = getAllPreprocessedFiles()
    perception = []
    imagrination = []
    for filepath in filepaths:
        perception_, imagination_ = getPerceptionImaginationDataFromSubject(filepath)
        perception.append(perception_)
        imagrination.append(imagination_)
    pdb.set_trace(0)

    perception = np.array(perception)
    imagrination = np.array(imagrination)
    
    perception = perception.reshape(-1, perception.shape[2], perception.shape[3])
    imagrination = imagrination.reshape(-1, imagrination.shape[2], imagrination.shape[3])


getAllPerceptionAndImaginationDataFromAllSubjects()