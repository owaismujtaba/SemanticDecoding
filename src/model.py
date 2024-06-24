import math
import torch
import torch.nn as nn

from data_loader import createDataLoaders
import config
import pdb


class PositionalEncoding(nn.Module):
    def __init__(self, dModel, dropout=0.1, maxLen=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, inputDim=config.inputDim, numClasses=config.numClasses, 
                 seqLength=config.seqLength, dModel=config.dModel, 
                 nHead=config.nHead, numEncoderLayers=config.numEncoderLayers, 
                 dimFeedforward=config.dimFeedforward, dropout=config.dropout
        ):
        super(TransformerClassifier, self).__init__()
        
        self.modelType = 'Transformer'
        self.dModel = dModel

        self.inputLinear = nn.Linear(inputDim, dModel)
        self.posEncoder = PositionalEncoding(dModel, dropout)
        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dModel, nHead, dimFeedforward, dropout), numEncoderLayers)
        self.fc = nn.Linear(dModel * seqLength, numClasses)
        
    def forward(self, src):
        batchSize, seqLength, inputDim = src.shape
        src = self.inputLinear(src)  
        src = self.posEncoder(src)   
        src = src.permute(1, 0, 2)  
        output = self.transformerEncoder(src) 
        output = output.permute(1, 0, 2).contiguous().view(batchSize, -1) 
        output = self.fc(output) 
        return output


class CNNModel(nn.Module):
    def __init__(self, numClasses):
        super(CNNModel, self).__init__()
        self.numClasses = numClasses
        self.conv1 = nn.Conv2d(in_channels=124, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(32 * 32 * 32, 512)  
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, self.numClasses)  

    def forward(self, x):
        #pdb.set_trace()
        x = x.view(-1, x.shape[1], 1,  x.shape[2])
        x = self.conv1(x)
        x = self.relu1(x)
        
        #x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        #x = self.pool2(x)
        
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])  
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x


class BiLSTMEEGClassifier(nn.Module):
    def __init__(self, inputDim=config.inputDim, hiddenDim=config.seqLength, nLayers=4):
        super(BiLSTMEEGClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=inputDim,
            hidden_size=hiddenDim,
            num_layers=nLayers, 
            bidirectional=True, 
            batch_first=True
        )
        self.fc = nn.Linear(hiddenDim * 2, config.numClasses)  # hidden_dim * 2 for bidirectional
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        #pdb.set_trace()
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out