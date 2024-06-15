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




