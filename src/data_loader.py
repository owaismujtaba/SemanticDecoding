import os
import numpy as np
import config
import pdb

class NumpyDataLoader:
    def __init__(self, rootDir, batchSize=32, shuffle=True):
        self.rootDir = rootDir
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.filePaths, self.labels = self.loadFilePathsAndLabels()
        self.numSamples = len(self.filePaths)
        self.onEpochEnd()
    
    def loadFilePathsAndLabels(self):
        filePaths = []
        labels = []
        classFolders = os.listdir(self.rootDir)
        for label, classFolder in enumerate(classFolders):
            classPath = os.path.join(self.rootDir, classFolder)
            if os.path.isdir(classPath):
                classFiles = os.listdir(classPath)
                for fileName in classFiles:
                    if fileName.endswith('.npy'):
                        filePaths.append(os.path.join(classPath, fileName))
                        labels.append(label)
        return filePaths, labels
    
    def onEpochEnd(self):
        self.indices = np.arange(self.numSamples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.floor(self.numSamples / self.batchSize))
    
    def __getitem__(self, index):
        batchIndices = self.indices[index * self.batchSize:(index + 1) * self.batchSize]
        batchPaths = [self.filePaths[i] for i in batchIndices]
        batchLabels = [self.labels[i] for i in batchIndices]
        batchData = [np.load(path)[:,config.startIndex:config.endIndex] for path in batchPaths]
        
        return np.array(batchData), np.array(batchLabels)

