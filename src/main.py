from data_loader import createDataLoaders
from model import TransformerClassifier, CNNModel, BiLSTMEEGClassifier
from trainner import trainModel
import config
from data_utils import SemanticSegmentation
from utils import loadTrainedModel
from eval import classificationReport
from pathlib import Path
from analysis import loadSementiciWiseData
import pdb


if __name__ == "__main__":

    if config.analysis:
        loadSementiciWiseData(activity='Perception', modality='Text')


    if config.segmentDataBasedOnSemantics:
        segmentData = SemanticSegmentation(scaling=True)
        segmentData.segmentFiles()


    if config.train:
        trainLoader, valLoader, _ = createDataLoaders(rootDir=config.scaledSementicDataDir)
        model = BiLSTMEEGClassifier()
        #model = CNNModel(config.numClasses)
        trainedModel = trainModel(model, trainLoader, valLoader, numEpochs=config.epochs, learningRate=0.001)

    if config.eval:
        trainedModelDir = config.trainedModelDir
        modelName = config.trainedModelName
        modelPath = Path(trainedModelDir, modelName)
        model = loadTrainedModel(modelPath)
        classificationReport(model)
        
