import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter

import config
import pdb



class NumpyDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        """
        Args:
            rootDir (string): Directory with all the class folders containing .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rootDir = rootDir
        self.transform = transform
        self.filePaths = []
        self.labels = []
        self.classNames = sorted(os.listdir(rootDir))
        
        self.classToIdx = {clsName: idx for idx, clsName in enumerate(self.classNames)}
        self.classCounts = {clsName: 0 for clsName in self.classNames}
        
        for clsName in self.classNames:
            clsFolder = os.path.join(rootDir, clsName)
            for fileName in os.listdir(clsFolder):
                if fileName.endswith('.npy'):
                    filePath = os.path.join(clsFolder, fileName)
                    self.filePaths.append(filePath)
                    self.labels.append(self.classToIdx[clsName])

    
        classCounts = Counter(self.labels)
        nSamples = len(self.labels)
        nClasses = len(classCounts)
        classWeights = [nSamples/(nClasses*classCounts[label]) for label in self.labels]
        self.sampler = WeightedRandomSampler(weights=classWeights, num_samples=len(classWeights), replacement=True)



    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filePath = self.filePaths[idx]
        sample = np.load(filePath)[:, config.startIndex:config.endIndex]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


def createDataLoaders(rootDir=config.trainDataDir, 
                      batchSize=config.batchSize, 
                      valSplit=0.2, 
                      transform=None,
                      numWorkers=1
                ):
    """
    Creates training and validation data loaders.

    Args:
        rootDir (str): Directory with all the class folders containing .npy files.
        batchSize (int): How many samples per batch to load.
        valSplit (float): Proportion of the dataset to include in the validation split.
        transform (callable, optional): Optional transform to be applied on a sample.
        numWorkers (int): How many subprocesses to use for data loading.

    Returns:
        tuple: Training DataLoader, Validation DataLoader
    """
    dataset = NumpyDataset(rootDir=rootDir, transform=transform)
    valLen = int(len(dataset) * valSplit)
    trainLen = len(dataset) - valLen
    trainDataset, valDataset = random_split(dataset, [trainLen, valLen])

    trainSampler = trainDataset.sampler if hasattr(trainDataset, 'sampler') else None
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, sampler=trainSampler, num_workers=numWorkers)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    
    return trainLoader, valLoader
