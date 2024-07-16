import numpy as np
import pandas as pd
import os
from pathlib import Path
import joblib
import pdb
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model

import src.config as config

def loadAllTrainedModels(decoding='PerceptionImagination'):
    modelsDir = Path(config.trainedModelsDir, decoding)
    models = os.listdir(modelsDir)
    models = [item for item in models if not item.endswith('.csv')]
    
    loadedModels = []
    modelNames = []
    for model in models:
        modelPath = Path(modelsDir, model)
        modelNames.append(model.split('.')[0])
        if model.endswith('.pkl'):
            loadedModels.append(loadModel(modelPath, type='.pkl'))
        else:
            loadedModels.append(loadModel(modelPath))

    return loadedModels, modelNames

def loadModel(modelPath, type='Tensor'):
    if type != 'Tensor':
        model = joblib.load(modelPath)
    else:
        model = load_model(modelPath)
    return model

def classificationReport(model, xTest, yTest):
    print('Classification Report')
    predictions = model.predict(xTest)  
    if len(predictions.shape) >1:
        predictions = np.argmax(predictions, axis=1)
    report = classification_report(yTest, predictions, output_dict=True)
    return report
            
def getIndividualSpecificClassificationReport(
        model, xTest, yTest, 
        subjectIds, sessionIds, 
        testSizes
    ):
    start = 0
    support = []
    weightedPrecision = []
    weightedRecall = []
    weightedF1Score = []
    accuracy = []
    
    
    for index in range(len(testSizes)):
        end = start + testSizes[index]

        X = xTest[start:end]
        y = yTest[start:end]
        start = end       

        report = classificationReport(model, X, y)
        
        
        weightedPrecision.append(report['weighted avg']['precision'])
        weightedRecall.append(report['weighted avg']['recall'] * report['weighted avg']['support'])
        weightedF1Score.append(report['weighted avg']['f1-score'] * report['weighted avg']['support'])
        support.append(report['weighted avg']['support'])
        accuracy.append(report['accuracy'])


    overallReport = {
        'subjectId':subjectIds,
        'sessionId':sessionIds,
        'precision': weightedPrecision,
        'recall': weightedRecall,
        'f1-score': weightedF1Score,
        'support': support,
        'accuracy':accuracy
    }
    
    overallReport = pd.DataFrame(overallReport)
    return overallReport

        
    

    
    
