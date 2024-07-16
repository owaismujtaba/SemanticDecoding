from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from pathlib import Path
import os
import pandas as pd
import pdb

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import DepthwiseConv2D, Activation, LSTM, Bidirectional
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D 
from tensorflow.keras.layers import SpatialDropout2D, Dropout, ReLU, Add
from tensorflow.keras.layers import  SeparableConv2D, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.activations import swish

import src.config as config
from src.utils import saveTensorFlowModel, getCSPFeatures

class EEGNetBasedModel:
    def __init__(self, numClasses, chans=124, samples=1059, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, normRate=0.25, dropoutType='Dropout'):
        print('*************Buidling EEGNet Based Architecture*****************')
        self.nbClasses = numClasses
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.normRate = normRate
        self.dropoutType = dropoutType

        if self.dropoutType == 'SpatialDropout2D':
            self.dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            self.dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

    def buildModel(self):
        input1 = Input(shape=(self.chans, self.samples, 1))

        # Block 1
        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', use_bias=False)(input1)
        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', use_bias=False)(block1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=self.D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = self.dropoutType(self.dropoutRate)(block1)

        # Block 2
        block2 = SeparableConv2D(self.F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = self.dropoutType(self.dropoutRate)(block2)

        # Flatten and Dense layers
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(self.nbClasses, name='dense', kernel_constraint=max_norm(self.normRate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        return Model(inputs=input1, outputs=softmax)

class EnhancedEEGNet:
    def __init__(self, numClasses, chans=124, samples=1059, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, normRate=0.25, dropoutType='Dropout',
                 kernel_size_block1=(1, 64), strides_block1=(1, 4),
                 kernel_size_block2=(1, 16), strides_block2=(1, 8)):
        print('*************Building Enhanced EEGNet Architecture*****************')
        self.nbClasses = numClasses
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.normRate = normRate
        self.dropoutType = dropoutType
        self.kernel_size_block1 = kernel_size_block1
        self.strides_block1 = strides_block1
        self.kernel_size_block2 = kernel_size_block2
        self.strides_block2 = strides_block2

        if self.dropoutType == 'SpatialDropout2D':
            self.dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            self.dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

    def buildModel(self):
        input1 = Input(shape=(self.chans, self.samples, 1))

        # Block 1
        block1 = Conv2D(self.F1, self.kernel_size_block1, padding='same', use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=self.D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation(swish)(block1)
        block1 = AveragePooling2D(self.strides_block1)(block1)
        block1 = self.dropoutType(self.dropoutRate)(block1)

        # Block 2 with Residual Connection
        block2 = SeparableConv2D(self.F2, self.kernel_size_block2, use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation(swish)(block2)
        block2 = AveragePooling2D(self.strides_block2)(block2)
        block2 = self.dropoutType(self.dropoutRate)(block2)

        # Adding residual connection with shape matching
        residual = Conv2D(self.F2, (1, 1), padding='same', use_bias=False, strides=self.strides_block2)(block1)
        residual = BatchNormalization()(residual)
        block2 = Add()([block2, residual])

        # Flatten and Dense layers
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(self.nbClasses, name='dense', kernel_constraint=max_norm(self.normRate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        return Model(inputs=input1, outputs=softmax)


class EnhancedCNNModel1(tf.keras.Model):
    def __init__(self, numClasses):
        super(EnhancedCNNModel1, self).__init__()
        self.conv1 = Conv2D(
            32, kernel_size=(3, 3), strides=(1, 1), 
            padding='same', input_shape=(124, 1059, 1)
        )
        self.bn1 =  BatchNormalization()
        self.relu1 = ReLU()
        self.pool1 =  MaxPooling2D(pool_size=(2, 2))
        self.dropout1 =  Dropout(0.25)
        
        self.conv2 =  Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2 =  BatchNormalization()
        self.relu2 =  ReLU()
        self.pool2 =  MaxPooling2D(pool_size=(2, 2))
        self.dropout2 =  Dropout(0.25)
        
        self.conv3 =  Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn3 =  BatchNormalization()
        self.relu3 =  ReLU()
        self.pool3 =  MaxPooling2D(pool_size=(2, 2))
        self.dropout3 =  Dropout(0.25)
        
        self.flatten =  Flatten()
        self.fc1 =  Dense(512)
        self.bn4 =  BatchNormalization()
        self.relu4 =  ReLU()
        self.dropout4 =  Dropout(0.5)
        
        self.fc2 =  Dense(numClasses)
        
    def call(self, inputs):
        print(inputs.shape)
        #x = tf.reshape(inputs, [inputs.shape[0], 124,1059,1])
        pdb.set_trace()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

class DeepConvNet:
    def __init__(self, 
                 numClasses, chans=124, 
                 samples=1000, dropoutRate=0.5
        ):
        print('*************Buidling DeepConvNet Architecture*****************')
        self.nbClasses = numClasses
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate

    def buildModel(self):
        inputMain = Input((self.chans, self.samples, 1))
        block1 = Conv2D(25, (1, 5), input_shape=(self.chans, self.samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(inputMain)
        block1 = Conv2D(25, (self.chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1 = Dropout(self.dropoutRate)(block1)

        block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2 = Dropout(self.dropoutRate)(block2)

        block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(self.dropoutRate)(block3)

        block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4 = Dropout(self.dropoutRate)(block4)

        flatten = Flatten()(block4)
        dense = Dense(self.nbClasses, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return Model(inputs=inputMain, outputs=softmax)

class EegNetSsvepN:
    def __init__(self, numClasses=12, numChannels=124, 
                 samples=1000, dropoutRate=0.5, kernLength=256, 
                 F1=96, D=1, F2=96, dropoutType='Dropout'
        ):
        print('*************Buidling EEGNetSSVEPN Architecture*****************')
        self.numClasses = numClasses
        self.numChannels = numChannels
        self.numTimepoints = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutType = dropoutType

    def buildModel(self):
        if self.dropoutType == 'SpatialDropout2D':
            dropoutLayer = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            dropoutLayer = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
        
        input1 = Input(shape=(self.numChannels, self.numTimepoints, 1))

        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', input_shape=(self.numChannels, self.numTimepoints, 1), use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.numChannels, 1), use_bias=False, depth_multiplier=self.D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutLayer(self.dropoutRate)(block1)
        
        block2 = SeparableConv2D(self.F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutLayer(self.dropoutRate)(block2)
        
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(self.numClasses, name='dense')(flatten)
        softmax = Activation('softmax', name='softmax')(dense)
        
        return Model(inputs=input1, outputs=softmax)

class EEGNet:
    def __init__(self, numClasses, chans=124, samples=1000, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, normRate=0.25, dropoutType='Dropout'):
        print('*************Buidling EEGNet Architecture*****************')
        self.nbClasses = numClasses
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.normRate = normRate
        self.dropoutType = dropoutType

        if self.dropoutType == 'SpatialDropout2D':
            self.dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            self.dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

    def buildModel(self):
        input1 = Input(shape=(self.chans, self.samples, 1))

        # Block 1
        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=self.D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = self.dropoutType(self.dropoutRate)(block1)

        # Block 2
        block2 = SeparableConv2D(self.F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = self.dropoutType(self.dropoutRate)(block2)

        # Flatten and Dense layers
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(self.nbClasses, name='dense', kernel_constraint=max_norm(self.normRate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        return Model(inputs=input1, outputs=softmax)

class XGBoostModel:
    def __init__(self, numClasses, taskType):
        print('*******************************************')
        print('***************** XGB Model ***************')
        self.name = "XGB"
        self.destinationDir = Path(config.trainedModelsDir, taskType)
        self.numClasses = numClasses
        self.model = None
        self.bestModel = None
        self.bestAccuracy = 0
        self.report = None
        self.bestModelName = None

        
    def hyperParameterTunning(self, xTrain, yTrain, xTest, yTest):
        xTrain = xTrain.reshape(xTrain.shape[0], -1)
        xTest = xTest.reshape(xTest.shape[0], -1)
        
        print(f'xTrain:{xTrain.shape}, xTest:{xTest.shape}')
        
        model = xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=self.numClasses, 
            seed=42,
            nthread=config.numJobs
        )
        params = {
            'maxDepth': [10, 20, 30],
            'nEstimators': [50, 100, 200]
        }
        for maxDepth in params['maxDepth']:
            for nEstimators in params['nEstimators']:
                modelName = f'max_depth: {maxDepth} nEstimators:{nEstimators}'
                print(modelName)
                          
                model.set_params(
                    max_depth=maxDepth, 
                    n_estimators=nEstimators
                )
                model.fit(xTrain, yTrain)
                    
                yPred = model.predict(xTest)
                accuracy = accuracy_score(yTest, yPred)
                if accuracy > self.bestAccuracy:
                    self.bestAccuracy = accuracy
                    self.bestModel = model
                    self.report = classification_report(yTest, yPred, output_dict=True)
                    self.bestModelName = model
        os.makedirs(self.destinationDir, exist_ok=True)
        modelNameWithPath = Path(self.destinationDir, f"{self.name}.pkl")
        saveTensorFlowModel(model, modelNameWithPath, "pickle")
        bestReport = self.report
        bestReport = pd.DataFrame(bestReport).T
        bestReport.to_csv(Path(self.destinationDir, f'{self.name}.csv'))
        self.bestModelName = modelName

class SVCModel:
    def __init__(self, numClasses, taskType):
        print('*******************************************')
        print('***************** SVC Model ***************')
        self.name = "SVC"
        self.destinationDir = Path(config.trainedModelsDir, taskType)
        self.numClasses = numClasses
        self.model = None
        self.bestModel = None
        self.bestAccuracy = 0
        self.report = None
        self.bestModelName = None

    def hyperParameterTunning(self, xTrain, yTrain, xTest, yTest):
        xTrain = xTrain.reshape(xTrain.shape[0], -1)
        xTest = xTest.reshape(xTest.shape[0], -1)
        print(f'xTrain:{xTrain.shape}, xTest:{xTest.shape}')
        paramGrid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
        for c in paramGrid['C']:
            for gamma in paramGrid['gamma']:
                for kernel in paramGrid['kernel']:                        
                        modelName = f'C: {c}_ gamme: {gamma}_kernal: {kernel}'
                        print('fitting', modelName)       
                model = SVC(
                        C = c,
                        gamma=gamma,
                        kernel=kernel,
                )
                model.fit(xTrain, yTrain)
                    
                yPred = model.predict(xTest)
                accuracy = accuracy_score(yTest, yPred)
                if accuracy > self.bestAccuracy:
                    self.bestAccuracy = accuracy
                    self.bestModel = self.model
                    self.report = classification_report(yTest, yPred, output_dict=True)
                    self.bestModelName = modelName
        os.makedirs(self.destinationDir, exist_ok=True)
        modelNameWithPath = Path(self.destinationDir, f"{self.name}.pkl")
        saveTensorFlowModel(model, modelNameWithPath, "pickle")
        bestReport = self.report
        bestReport = pd.DataFrame(bestReport).T
        bestReport.to_csv(Path(self.destinationDir, f'{self.name}.csv'))
        self.bestModelName = modelName

class RandomForestModel:
    def __init__(self, numClasses, taskType):
        print('*******************************************')
        print('************* Random Forest Model ***********')
        self.name = "RF"
        self.destinationDir = Path(config.trainedModelsDir, taskType)
        self.numClasses = numClasses
        self.model = None
        self.bestModel = None
        self.bestAccuracy = 0
        self.report = None
        self.bestModelName = None
    

    def hyperParameterTunning(self,xTrain, yTrain, xTest, yTest):
        xTrain = xTrain.reshape(xTrain.shape[0], -1)
        xTest = xTest.reshape(xTest.shape[0], -1)
        
        paramGrid = {
            'n_estimators': [20, 30, 50, 100, 500, 1000],
            'max_depth': [10, 20, 30],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [10, 15, 20, 25, 30]
        }
        for nEstimators in paramGrid['n_estimators']:
            for maxDepth in paramGrid['max_depth']:
                for maxFeatures in paramGrid['max_features']:
                    for minSamplesSplit in paramGrid['min_samples_split']:
                        modelName = f'{nEstimators}_{maxDepth}_{maxFeatures}_{minSamplesSplit}'
                        print(f'Fitting {modelName}')
                        model = RandomForestClassifier(
                            n_estimators=nEstimators,
                            max_depth=maxDepth,
                            min_samples_split=minSamplesSplit,
                            max_features=maxFeatures,
                            n_jobs=config.numJobs
                        )
                        model.fit(xTrain, yTrain)
                        yPred = model.predict(xTest)
                        
                        accuracy = accuracy_score(yTest, yPred)
                        if accuracy > self.bestAccuracy:
                            self.bestAccuracy = accuracy
                            self.bestModel = self.model
                            self.report = classification_report(yTest, yPred, output_dict=True)
                            self.bestModelName = modelName
                    
                
            
            
                
        os.makedirs(self.destinationDir, exist_ok=True)
        modelNameWithPath = Path(self.destinationDir, f"{self.name}.pkl")
        saveTensorFlowModel(model, modelNameWithPath, "pickle")

        bestReport = self.report
        bestReport = pd.DataFrame(bestReport).T
        bestReport.to_csv(Path(self.destinationDir, f'{self.name}.csv'))

