from data_loader import createDataLoaders
from model import TransformerClassifier
from trainner import trainModel
import config
from data_utils import SemanticSegmentation
from utils import loadTrainedModel
from eval import classificationReport
from pathlib import Path
import pdb


if __name__ == "__main__":

    if config.segmentDataBasedOnSemantics:
        segmentData = SemanticSegmentation()
        segmentData.segmentFiles()


    if config.train:
        trainLoader, valLoader, _ = createDataLoaders()
        model = TransformerClassifier(config.inputDim, config.numClasses)
        trainedModel = trainModel(model, trainLoader, valLoader, numEpochs=config.epochs, learningRate=0.001)

    if config.eval:
        trainedModelDir = config.trainedModelDir
        modelName = config.trainedModelName
        modelPath = Path(trainedModelDir, modelName)
        model = loadTrainedModel(modelPath)
        classificationReport(model)
        
