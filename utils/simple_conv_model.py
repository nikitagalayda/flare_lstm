import numpy as np
from numpy import float32
import warnings
import os
import glob
import tensorflow as tf
import cv2
from sklearn import utils
from sklearn import preprocessing
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import *
from tensorflow.keras.utils import *
from tensorflow.keras import layers

from data_augmentation import *

def get_simple_conv_model():
    inp = Input(shape=(64, 64, 1))
    x = Conv2D(filters=48, kernel_size=4)(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(filters=24, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(filters=12, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, x)
    
    return model