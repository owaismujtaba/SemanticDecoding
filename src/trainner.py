from data_loader import NumpyDataLoader
from model import EEGTransformer
import tensorflow as tf
import config

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    outputs = Dense(ff_dim, activation='relu')(attention)
    outputs = Dense(inputs.shape[-1])(outputs)  
    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)
    
    return outputs

num_channels = 124  
num_time_points = 1024  
num_classes = 3  
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)



def build_transformer_model(num_channels, num_time_points, num_classes):
    inputs = Input(shape=(num_channels, num_time_points))

    transformer_block = inputs  
    for _ in range(2): 
        transformer_block = transformer_encoder(transformer_block, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)
    pool = GlobalAveragePooling1D()(transformer_block)
    dropout = Dropout(0.5)(pool)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_transformer_model(num_channels, num_time_points, num_classes)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
dataLoader = NumpyDataLoader(config.trainDataDir, config.batchSize)
for epoch in range(100):
    for batch_data, batch_labels in dataLoader:
        batch_data_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            predictions = model(batch_data_tf)
            loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels_tf, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f'Epoch {epoch + 1}/{100}, Loss: {loss.numpy()}')


def trainer():
    numLayers = config.numLayers
    embedDim = config.embedDim
    numHeads = config.numHeads
    dff = config.dff
    inputVocabSize = config.inputVocabSize
    maximumPositionEncoding = config.maximumPositionEncoding
    numClasses = config.numClasses

    dataLoader = NumpyDataLoader(config.trainDataDir, config.batchSize)

    model = EEGTransformer(numLayers, embedDim, numHeads,dff,inputVocabSize, maximumPositionEncoding, numClasses)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    epochs = config.epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} /{epochs}')
        for dataBatch, labelsBatch in dataLoader:
            print(dataBatch.shape, labelsBatch.shape)
            dataBatch = tf.convert_to_tensor(dataBatch, dtype=tf.float32)
            labelsBatch = tf.convert_to_tensor(labelsBatch, dtype=tf.int64)
            model.train_on_batch(dataBatch, labelsBatch)