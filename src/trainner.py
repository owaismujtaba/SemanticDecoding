from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from utils import saveTrainedModel

def trainModel(model, trainLoader, valLoader, numEpochs=25, learningRate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        with tqdm(total=len(trainLoader), desc=f"Epoch {epoch+1}/{numEpochs}", unit="batch") as pbar:
            for inputs, labels in trainLoader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
                pbar.set_postfix({'Training Loss': runningLoss / (len(trainLoader) * trainLoader.batch_size)})
                pbar.update()
                
        model.eval()
        valLoss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f"Epoch {epoch+1}/{numEpochs}, Training Loss: {runningLoss/len(trainLoader)}, Validation Loss: {valLoss/len(valLoader)}, Validation Accuracy: {100 * correct / total}%")

    saveTrainedModel(model)
