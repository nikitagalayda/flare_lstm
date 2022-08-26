import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def SingleConvLSTMModel(batch_size, image_size, sequence_length):
    inp = Input(shape=(sequence_length, image_size, image_size, 1))
    print(inp.shape)
    x = Bidirectional(ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    ))(inp)
    x = BatchNormalization()(x)
    x = Bidirectional(ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    ))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(ConvLSTM2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    ))(x)

    model = Model(inp, x)
    
    return model