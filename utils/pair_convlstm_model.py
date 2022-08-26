import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from single_convlstm_model import *

def PairConvLSTMModel(batch_size, image_size, sequence_length, num_classes=3):
    inp = Input(shape=(sequence_length, image_size, image_size, 1))
    print(inp.shape)
    full_inp = tf.convert_to_tensor(inp[:, :(sequence_length//2), :, :, :])
    cutout_inp = tf.convert_to_tensor(inp[:, (sequence_length//2):, :, :, :])
    
    full_features = SingleConvLSTMModel(batch_size, image_size, sequence_length)(inp)
    cutout_features = SingleConvLSTMModel(batch_size, image_size, sequence_length)(cutout_inp)
    x = concatenate([full_features, cutout_features])
    x = GlobalMaxPooling3D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inp, x)
    
    return model