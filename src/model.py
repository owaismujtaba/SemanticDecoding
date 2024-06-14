import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pdb
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embedDim, numHeads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.projectionDim = embedDim // numHeads
        assert embedDim % numHeads == 0

        self.queryDense = layers.Dense(embedDim)
        self.keyDense = layers.Dense(embedDim)
        self.valueDense = layers.Dense(embedDim)
        self.combineHeads = layers.Dense(embedDim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dimKey = tf.cast(tf.shape(key)[-1], tf.float32)
        scaledScore = score / tf.math.sqrt(dimKey)
        weights = tf.nn.softmax(scaledScore, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separateHeads(self, x, batchSize):
        x = tf.reshape(x, (batchSize, -1, self.numHeads, self.projectionDim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batchSize = tf.shape(inputs)[0]
        query = self.queryDense(inputs)
        key = self.keyDense(inputs)
        value = self.valueDense(inputs)

        query = self.separateHeads(query, batchSize)
        key = self.separateHeads(key, batchSize)
        value = self.separateHeads(value, batchSize)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concatAttention = tf.reshape(attention, (batchSize, -1, self.embedDim))
        output = self.combineHeads(concatAttention)
        return output

class FeedForwardNetwork(layers.Layer):
    def __init__(self, dModel, dff):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(dModel)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class EncoderLayer(layers.Layer):
    def __init__(self, embedDim, numHeads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadSelfAttention(embedDim, numHeads)
        self.ffn = FeedForwardNetwork(embedDim, dff)

        self.layerNorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=None):
        attnOutput = self.mha(x)
        attnOutput = self.dropout1(attnOutput, training=training)
        out1 = self.layerNorm1(x + attnOutput)

        ffnOutput = self.ffn(out1)
        ffnOutput = self.dropout2(ffnOutput, training=training)
        out2 = self.layerNorm2(out1 + ffnOutput)

        return out2

class TransformerEncoder(layers.Layer):
    def __init__(self, numLayers, embedDim, numHeads, dff, inputVocabSize, maximumPositionEncoding, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.embedDim = embedDim
        self.numLayers = numLayers

        self.embedding = layers.Embedding(inputVocabSize, embedDim)
        self.posEncoding = self.positionalEncoding(maximumPositionEncoding, embedDim)

        self.encLayers = [EncoderLayer(embedDim, numHeads, dff, rate) for _ in range(numLayers)]
        self.dropout = layers.Dropout(rate)

    def positionalEncoding(self, position, dModel):
        angleRads = self.getAngles(np.arange(position)[:, np.newaxis], np.arange(dModel)[np.newaxis, :], dModel)
        angleRads[:, 0::2] = np.sin(angleRads[:, 0::2])
        angleRads[:, 1::2] = np.cos(angleRads[:, 1::2])
        posEncoding = angleRads[np.newaxis, ...]
        return tf.cast(posEncoding, dtype=tf.float32)

    def getAngles(self, pos, i, dModel):
        angleRates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dModel))
        return pos * angleRates

    def call(self, x, training=None):
        seqLen = tf.shape(x)[1]  # Assuming sequence length is 124
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedDim, tf.float32))
        pdb.set_trace()
        # Adjust positional encoding to match sequence length
        posEncoding = self.posEncoding[:, :seqLen, :]
        
        x += posEncoding  # Add positional encoding
        
        x = self.dropout(x, training=training)

        for i in range(self.numLayers):
            x = self.encLayers[i](x, training=training)

        return x

class EEGTransformer(Model):
    def __init__(self, numLayers, embedDim, numHeads, dff, inputVocabSize, maximumPositionEncoding, numClasses, rate=0.1):
        super(EEGTransformer, self).__init__()
        self.encoder = TransformerEncoder(numLayers, embedDim, numHeads, dff, inputVocabSize, maximumPositionEncoding, rate)
        self.globalAveragePooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(rate)
        self.finalLayer = layers.Dense(numClasses, activation='softmax')

    def call(self, x, training=True):  
        x = self.encoder(x, training=training) 
        x = self.globalAveragePooling(x)
        x = self.dropout(x, training=training)
        x = self.finalLayer(x)
        return x
