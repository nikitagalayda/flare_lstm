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

def CustomAlexNet():
    inp = Input(shape=(128, 128, 1))
    x = Conv2D(filters=96, kernel_size=(11, 11), activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3))(x)
    
    x = Conv2D(filters=256, kernel_size=(5, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3))(x)
    
    x = Conv2D(filters=384, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=384, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPool2D(pool_size=(3, 3))(x)
    
    x = Flatten()(x)
    
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, x)
    
    return model