from data_loader import createDataLoaders
import torch
import pdb
from sklearn.metrics import classification_report

def classificationReport(model):

    _, valLoader, idx = createDataLoaders()
    model.eval()
    with torch.no_grad():
        for inputs, labels in valLoader:        
            outputs = model(inputs)
            
            predictions = torch.argmax(outputs, 1)
            predictions = predictions.numpy().astype(int)
            labels = labels.numpy().astype(int)
            report = classification_report(labels, predictions, target_names=list(idx.keys()))
            print(report)
            


    
    
