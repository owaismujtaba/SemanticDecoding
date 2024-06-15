from data_loader import createDataLoaders
from model import TransformerClassifier
from trainner import trainModel


if __name__ == "__main__":
    trainLoader, valLoader = createDataLoaders()
    model = TransformerClassifier(1024, 3)
    trainedModel = trainModel(model, trainLoader, valLoader, numEpochs=25, learningRate=0.001)
