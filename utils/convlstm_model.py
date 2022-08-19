import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def SingleConvLSTMModel(batch_size):
    # Construct the input layer with no definite frame size.
    inp = Input(shape=(6, 64, 64, 1))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = Dense(256, activation='relu')(x)
    model = Model(inp, x)
    
    return model

def ConvLSTMModel(batch_size):
    inp = Input(shape=(12, 64, 64, 1))
    print(inp.shape)
    full_inp = tf.convert_to_tensor(inp[:, :6, :, :, :])
    cutout_inp = tf.convert_to_tensor(inp[:, 6:, :, :, :])
    
    x = SingleConvLSTMModel(batch_size)(full_inp)
    x = GlobalMaxPooling3D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inp, x)
    
    return model