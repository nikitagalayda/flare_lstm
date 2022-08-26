import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from single_convlstm_model import *

def ConvLSTMModelAllClass(batch_size, image_size, sequence_length, num_classes=3):
    inp = Input(shape=(sequence_length, image_size, image_size, 1))
    
    x = SingleConvLSTMModel(batch_size, image_size, sequence_length)(inp)
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(2, 2)
    )(x)
    x = GlobalMaxPooling3D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, x)
    
    return model