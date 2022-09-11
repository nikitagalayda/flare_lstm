#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import float32
import warnings
import os
import sys
import glob
import tensorflow as tf
import cv2
from sklearn import utils
from sklearn import preprocessing
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import *
from tensorflow.keras.utils import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append(os.path.join(Path.cwd(), 'utils'))
sys.path.append(os.path.join(Path.cwd(), 'data_generators'))
sys.path.append(os.path.join(Path.cwd(), 'models'))

from utils.im_utils import *
from utils.data_augmentation import *

from data_generators.FullImageBinaryGen import *
from data_generators.PairAllClassGen import *
from data_generators.FullImageAllClassGen import *
from data_generators.CutoutImageAllClassGen import *

from models.bidirectional_convlstm_model import *
from models.pair_convlstm_model import *


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:


FLARE_CLASS = 'ALL'
BEST_TRAINED_MODELS_DIR = './best_trained_models'

LSTM_CHECKPOINTS_DIR = './checkpoints/lstm_checkpoints'
RESNET_CHECKPOINTS_DIR = './checkpoints/resnet_checkpoints'

# AUG_TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_data_augmented/train'
# AUG_VAL_DATA_DIR = f'./data/{FLARE_CLASS}_data_augmented/val'
# AUG_TEST_DATA_DIR = f'./data/{FLARE_CLASS}_data_augmented/test'

TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_extended/train'
VAL_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_extended/val'
TEST_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_extended/test'

AUG_TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_extended_augmented/train'
AUG_VAL_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_extended_augmented/val'
AUG_TEST_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_extended_augmented/test'

AUG_PAIR_TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented_pair/train'
AUG_PAIR_VAL_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented_pair/val'
AUG_PAIR_TEST_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented_pair/test'

AUG_END_PAIR_TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_end_data_augmented_pair/train'
AUG_END_PAIR_VAL_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_end_data_augmented_pair/val'
AUG_END_PAIR_TEST_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_end_data_augmented_pair/test'

AUG_ALL_CLASS_TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented/train'
AUG_ALL_CLASS_VAL_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented/val'

AUG_ALL_CLASS_PRIOR_TRAIN_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented_prior/train/'
AUG_ALL_CLASS_PRIOR_VAL_DATA_DIR = f'./data/{FLARE_CLASS}_lstm_data_augmented_prior/val/'

# DATA_FEATURES_DIR = './data/data_features_simple'
# DATA_FEATURES_TRAIN_DIR = './data/data_features_simple/train'
# DATA_FEATURES_VAL_DIR = './data/data_features_simple/val'
# DATA_FEATURES_TEST_DIR = './data/data_features_simple/test'

LSTM_ALL_CLASS_PRIOR_DATA_DIR = f'./new_data/{FLARE_CLASS}_lstm_data_prior'
LSTM_ALL_CLASS_DURING_DATA_DIR = f'./new_data/{FLARE_CLASS}_lstm_data_during'
LSTM_ALL_CLASS_LATESTART_DATA_DIR = f'./new_data/{FLARE_CLASS}_lstm_data_latestart'


# In[4]:


def delete_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# In[5]:


def GetDataFolders(train_data_dir, val_data_dir):
    train_folders = []
    for subdir, dirs, files in os.walk(train_data_dir):
        for d in dirs:
            if d != 'positive' and d != 'negative' and d != '.ipynb_checkpoints':
                train_folders.append(os.path.join(subdir, d))
    train_folders = np.array(train_folders)

    val_folders = []
    for subdir, dirs, files in os.walk(val_data_dir):
         for d in dirs:
                if d != 'positive' and d != 'negative' and d != '.ipynb_checkpoints':
                    val_folders.append(os.path.join(subdir, d))
    val_folders = np.array(val_folders)
    
    return train_folders, val_folders


# In[6]:


def GetPairDataFolders(train_data_dir, val_data_dir):
    train_folders = set()
    for subdir, dirs, files in os.walk(train_data_dir):
        for f in files:
            flare_class = os.path.join(subdir, f).rsplit('/')[-5]
            # if flare_class == 'C':
            #     continue
            file_parent_path = os.path.join(subdir, f).rsplit('/', 2)[0]
            train_folders.add(file_parent_path)

    val_folders = set()
    for subdir, dirs, files in os.walk(val_data_dir):
        for f in files:
            flare_class = os.path.join(subdir, f).rsplit('/')[-5]
            # if flare_class == 'C':
            #     continue
            file_parent_path = os.path.join(subdir, f).rsplit('/', 2)[0]
            val_folders.add(file_parent_path)
    
    return list(train_folders), list(val_folders)


# In[7]:


def get_labels(generator, feature_extractor):
    labels = []

    for sample in generator:
        new_batch = []
        batch = sample[1]
        labels.append(batch)

    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2])
    
    return labels


# In[8]:


batch_size=128
num_classes=3
sequence_length=6
data_dir = 'ALL_lstm_data_nmx_prior_leftout2013_cadence6'
train_dir = os.path.join(f'./new_data/{data_dir}/', 'train')
val_dir = os.path.join(f'./new_data/{data_dir}/', 'val')
train_folders, val_folders = GetPairDataFolders(train_dir, val_dir)
traingen = FullImageAllClassGen(train_folders, batch_size=batch_size, image_size=64, num_classes=num_classes, sequence_length=sequence_length)
valgen = FullImageAllClassGen(val_folders, batch_size=26, image_size=64, num_classes=num_classes, sequence_length=sequence_length)


# In[9]:


model = ConvLSTMModelAllClass(batch_size, 64, sequence_length-1, num_classes)


# In[10]:


mc = ModelCheckpoint(f'{BEST_TRAINED_MODELS_DIR}/{data_dir}.h5', monitor='val_accuracy', save_best_only=True)


# In[11]:


callbacks_list = [mc]


# In[12]:


adam_fine = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.0002, amsgrad=False)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=1e-3,
decay_steps=10000,
decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(
    loss="categorical_crossentropy", optimizer=adam_fine, metrics=["accuracy"]
)


# In[13]:


epochs=200
history = model.fit(traingen, validation_data=valgen, epochs=epochs, callbacks=callbacks_list)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


model.save_weights(f'{LSTM_CHECKPOINTS_DIR}/{data_dir}')


# In[ ]:


# data_folder = './new_data/ALL_lstm_data_during_leftout2013/train/M/AIA20100807_1748_0094/0/full'
# paths = []
# for subdir, dirs, files in os.walk(data_folder):
#     for f in files:
#         paths.append(os.path.join(subdir, f))
# paths = sorted(paths)
# for p in paths:
#     print(p)

# fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# for idx, ax in enumerate(axes.flat):
#     ax.imshow(np.squeeze(preprocessing.normalize(np.load(paths[idx]))), cmap='jet')
#     ax.set_title(f"Frame {idx + 1}")
#     ax.axis("off")

# plt.show()

