#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from numpy import float32
import warnings
import os
import sys
import glob
import tensorflow as tf
import cv2
from sklearn import utils
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
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
import tensorflow_addons as tfa

sys.path.append(os.path.join(Path.cwd(), 'utils'))
sys.path.append(os.path.join(Path.cwd(), 'data_generators'))
sys.path.append(os.path.join(Path.cwd(), 'models'))

from utils.im_utils import *
from utils.data_augmentation import *
from utils.region_detector import *

from data_generators.FullImageBinaryGen import *
from data_generators.PairAllClassGen import *
from data_generators.FullImageAllClassGen import *
from data_generators.CutoutImageAllClassGen import *
from data_generators.FullImageSingleClassGen import *

from models.bidirectional_convlstm_model import *
from models.pair_convlstm_model import *


# In[23]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[24]:


FLARE_CLASS = 'ALL'
BEST_TRAINED_MODELS_DIR = './best_trained_models/'

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


# In[25]:


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


# In[26]:


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


# In[27]:


def GetFlaresDataFolders(train_data_dir, val_data_dir, flare_classes):
    train_folders = set()
    for subdir, dirs, files in os.walk(train_data_dir):
        for f in files:
            flare_class = os.path.join(subdir, f).rsplit('/')[-5]
            if flare_class not in flare_classes:
                continue
            file_parent_path = os.path.join(subdir, f).rsplit('/', 2)[0]
            train_folders.add(file_parent_path)

    val_folders = set()
    for subdir, dirs, files in os.walk(val_data_dir):
        for f in files:
            flare_class = os.path.join(subdir, f).rsplit('/')[-5]
            if flare_class not in flare_classes:
                continue
            file_parent_path = os.path.join(subdir, f).rsplit('/', 2)[0]
            val_folders.add(file_parent_path)
    
    return list(train_folders), list(val_folders)


# In[28]:


def GetSingleClassDataFolders(train_data_dir, val_data_dir, flare_class):
    train_folders = set()
    for subdir, dirs, files in os.walk(train_data_dir):
        for f in files:
            cur_class = os.path.join(subdir, f).rsplit('/')[-5]
            if cur_class != flare_class and cur_class != 'N':
                continue
            file_parent_path = os.path.join(subdir, f).rsplit('/', 2)[0]
            train_folders.add(file_parent_path)

    val_folders = set()
    for subdir, dirs, files in os.walk(val_data_dir):
        for f in files:
            cur_class = os.path.join(subdir, f).rsplit('/')[-5]
            if cur_class != flare_class and cur_class != 'N':
                continue
            file_parent_path = os.path.join(subdir, f).rsplit('/', 2)[0]
            val_folders.add(file_parent_path)
    
    return list(train_folders), list(val_folders)


# In[29]:


def get_labels(generator, feature_extractor):
    labels = []

    for sample in generator:
        new_batch = []
        batch = sample[1]
        labels.append(batch)

    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2])
    
    return labels


# In[30]:


# class TestImageAllClassGen(tf.keras.utils.Sequence):
#     def __init__(
#         self,
#         folder_paths,
#         batch_size,
#         shuffle=True,
#         image_size=64,
#         num_classes=3,
#         sequence_length=6,
#     ):

#         self.folder_paths = folder_paths.copy()
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.sequence_length = sequence_length
#         self.image_size = image_size

#         self.n = len(self.folder_paths)
#         self.n_category = num_classes

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.folder_paths)

#     def __getitem__(self, index):
#         batches = self.folder_paths[
#             index * self.batch_size : (index + 1) * self.batch_size
#         ]
#         X, y = self.__get_data(batches)
#         X = np.expand_dims(X, axis=4)
#         return X, y

#     def __len__(self):
#         return self.n // self.batch_size

#     def __get_input(self, folder):
#         images = []
#         for subdir, dirs, files in os.walk(folder):
#             for f in files:
#                 images.append(os.path.join(subdir, f))
#         images = sorted(images)
#         images = [np.load(x) for x in images[: self.sequence_length]]
#         images = [
#             abs(abs(images[x]) - abs(images[x - 1]))
#             for x in range(1, self.sequence_length)
#         ]
#         images = [
#             cv2.resize(
#                 x, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
#             )
#             for x in images
#         ]
#         images = np.array(images)
#         return images

#     def __get_output(self, path):
#         label = None
#         folder = path.rsplit("/")[-3]
#         if folder == "N":
#             label = 0
#         elif folder == 'C':
#             label = 1
#         elif folder == "M":
#             label = 2
#         elif folder == "X":
#             label = 3

#         one_hot_label = tf.one_hot(label, self.n_category)

#         return one_hot_label

#     def __get_data(self, batches):
#         X_batch = np.asarray([self.__get_input(x) for x in batches])
#         y_batch = np.asarray([self.__get_output(y) for y in batches])

#         return X_batch, y_batch


# In[31]:


class TestFullImageAllClassGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        folder_paths,
        batch_size,
        flare_classes,
        shuffle=True,
        image_size=64,
        sequence_length=6,
    ):

        self.folder_paths = folder_paths.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.n = len(self.folder_paths)
        self.n_category = len(flare_classes)
        self.flare_classes = flare_classes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.folder_paths)

    def __getitem__(self, index):
        # batches: batches of folder paths to data
        batches = self.folder_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X, y = self.__get_data(batches)
        X = np.expand_dims(X, axis=4)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def __get_input(self, folder):
        images = []
        for subdir, dirs, files in os.walk(folder):
            for f in files:
                images.append(os.path.join(subdir, f))
        images = sorted(images)
        images = [np.load(x) for x in images[:self.sequence_length]]
        images = [
            abs(abs(images[x]) - abs(images[x - 1]))
            for x in range(1, self.sequence_length-2)
        ]
        images = [
            cv2.resize(
                x, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            for x in images
        ]
        images = np.array(images)
        return images

    def __get_output(self, path):
        label = None
        folder = path.rsplit("/")[-3]
        folder_index_label = self.flare_classes.index(folder)

        one_hot_label = tf.one_hot(folder_index_label, self.n_category)

        return one_hot_label

    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(x) for x in batches])
        y_batch = np.asarray([self.__get_output(y) for y in batches])

        return X_batch, y_batch


# In[64]:


class TestPairAllClassGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        folder_paths,
        batch_size,
        flare_classes,
        shuffle=True,
        image_size=64,
        sequence_length=6,
    ):

        self.folder_paths = folder_paths.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.n = len(self.folder_paths)
        self.n_category = len(flare_classes)
        self.flare_classes = flare_classes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.folder_paths)

    def __getitem__(self, index):
        # batches: batches of folder paths to data
        batches = self.folder_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X, y = self.__get_data(batches)
        X = np.expand_dims(X, axis=4)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def __get_input(self, folder):
        images = []
        for subdir, dirs, files in os.walk(folder):
            for f in files:
                images.append(os.path.join(subdir, f))
        images = sorted(images)
        full_images = [np.load(x) for x in images[:self.sequence_length//2]]
        cutout_images = [np.load(x) for x in images[self.sequence_length//2:]]
        full_images = [abs(full_images[x]-full_images[x-1]) for x in range(1, self.sequence_length//2)]
        cutout_images = [abs(cutout_images[x]-cutout_images[x-1]) for x in range(1, self.sequence_length//2)]
        full_images = [cv2.resize(x, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA) for x in full_images]
        cutout_images = [cv2.resize(x, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA) for x in cutout_images]
        full_images  = np.array(full_images)
        cutout_images  = np.array(cutout_images)
        
        return np.concatenate([full_images, cutout_images])

    def __get_output(self, path):
        label = None
        folder = path.rsplit("/")[-3]
        folder_index_label = self.flare_classes.index(folder)

        one_hot_label = tf.one_hot(folder_index_label, self.n_category)

        return one_hot_label

    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(x) for x in batches])
        y_batch = np.asarray([self.__get_output(y) for y in batches])

        return X_batch, y_batch


# In[33]:


class TestFullImageBinaryGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        folder_paths,
        batch_size,
        flare_classes,
        shuffle=True,
        image_size=64,
        sequence_length=6,
    ):

        self.folder_paths = folder_paths.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.n = len(self.folder_paths)
        self.n_category = len(flare_classes)
        self.flare_classes = flare_classes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.folder_paths)

    def __getitem__(self, index):
        # batches: batches of folder paths to data
        batches = self.folder_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X, y = self.__get_data(batches)
        X = np.expand_dims(X, axis=4)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def __get_input(self, folder):
        images = []
        for subdir, dirs, files in os.walk(folder):
            for f in files:
                images.append(os.path.join(subdir, f))
        images = sorted(images)
        images = [np.load(x) for x in images[:self.sequence_length]]
        images = [
            abs(abs(images[x]) - abs(images[x - 1]))
            for x in range(1, self.sequence_length)
        ]
        images = [
            cv2.resize(
                x, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            for x in images
        ]
        images = np.array(images)
        return images

    def __get_output(self, path):
        label = None
        folder = path.rsplit("/")[-3]
        
        if folder == 'H':
            label = 0
        elif folder == 'M' or folder == 'X':
            label = 1
            
        return tf.one_hot(label, 2)

    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(x) for x in batches])
        y_batch = np.asarray([self.__get_output(y) for y in batches])

        return X_batch, y_batch


# In[34]:


# paths = []
# for subdir, dirs, files in os.walk('./new_data/cadence6_frame6/val/M/AIA20131219_2306_0094/0/full'):
#     for f in files:
#         paths.append(os.path.join(subdir, f))
# paths = sorted(paths)

# ims = [np.load(p) for p in paths]
# diff_ims = abs(np.array([abs(ims[i])-abs(ims[i-1]) for i in range(1, len(ims))]))


# In[35]:


# fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# for idx, ax in enumerate(axes.flat):
#     ax.imshow(ims[idx], cmap='jet')
#     ax.set_title(f"Frame {idx + 1}")
#     ax.axis("off")

# plt.show()


# In[36]:


# paths = []
# for subdir, dirs, files in os.walk('./new_data/cadence6_frame6/val/N/AIA20130113_0515_0094/0/full'):
#     for f in files:
#         paths.append(os.path.join(subdir, f))
# paths = sorted(paths)

# ims = [np.load(p) for p in paths]
# diff_ims = abs(np.array([abs(ims[i])-abs(ims[i-1]) for i in range(1, len(ims))]))
# last_im = ims[-1]
# last_im.max()


# In[55]:


flare_classes=['H', 'M', 'X']
batch_size=64
num_classes=len(flare_classes)
sequence_length=12
data_dir = 'cadence6_frame6'
output_name = f"{''.join(flare_classes)}_{data_dir}"
train_dir = os.path.join(f'./new_data/{data_dir}/', 'train')
val_dir = os.path.join(f'./new_data/{data_dir}/', 'val')
train_folders, val_folders = GetFlaresDataFolders(train_dir, val_dir, flare_classes)


# In[65]:


traingen = TestPairAllClassGen(
    train_folders, 
    batch_size, 
    flare_classes,
    image_size=64, 
    sequence_length=sequence_length
)

valgen = TestPairAllClassGen(
    val_folders, 
    batch_size, 
    flare_classes,
    image_size=64, 
    sequence_length=sequence_length
)


# In[39]:


# model = tf.keras.models.load_model(f'./best_trained_models/HMX_cadence6_frame6_pre_3frames_pre.h5')

# true_vals = [x[1] for x in valgen]
# true_vals = np.concatenate(true_vals)
# true_labels = [x.argmax() for x in true_vals]

# preds = model.predict(valgen)
# pred_labels = [x.argmax() for x in preds]

# c = 0
# for i, l in enumerate(pred_labels):
#      if l == true_labels[i]:
#             c+=1
# print(c/len(pred_labels))

# tf.math.confusion_matrix(
#     true_labels,
#     pred_labels,
#     num_classes=num_classes,
# )


# In[40]:


# classes = {}
# valid_classes = ['N', 'H', 'C', 'M', 'X']

# for folder in val_folders:
#     for subdir, dirs, files in os.walk(folder):
#         flare_class = subdir.rsplit('/')[-3]
#         if flare_class in valid_classes:
#             if flare_class in classes:
#                 classes[flare_class]+=1
#             else:
#                 classes[flare_class]=1


# In[41]:


# traingen
# N tuples representing N batches
# each tuple is value, label


# In[67]:


model = PairConvLSTMModel(batch_size, 64, sequence_length-2, num_classes)


# In[68]:


model.summary()


# In[69]:


mc = ModelCheckpoint(f'{BEST_TRAINED_MODELS_DIR}/{output_name}_binary.h5', monitor='val_loss', save_best_only=True)


# In[70]:


callbacks_list = [mc]
metrics = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tfa.metrics.F1Score(num_classes=num_classes)
]


# In[71]:


adam_fine = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.0002, amsgrad=False)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=1e-3,
decay_steps=10000,
decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(
    loss="categorical_crossentropy", optimizer=adam_fine, metrics=metrics
)


# In[ ]:


epochs=10
history = model.fit(traingen, validation_data=valgen, epochs=epochs, callbacks=callbacks_list)


# In[ ]:


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


# In[ ]:


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


# In[ ]:


model.save_weights(f'{LSTM_CHECKPOINTS_DIR}/{output_name}')


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


# In[ ]:


flare_before = [
    '../data_94/2013/05/03/AIA20130503_0424_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0430_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0436_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0442_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0448_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0454_0094.npz',
]

flare_start = [
    '../data_94/2013/05/03/AIA20130503_0512_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0518_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0524_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0530_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0536_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0542_0094.npz',
]

flare_pre = [
    '../data_94/2013/05/03/AIA20130503_0442_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0448_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0454_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0500_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0506_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0512_0094.npz',
]

flare_mid = [
    '../data_94/2013/05/03/AIA20130503_0500_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0506_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0512_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0518_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0524_0094.npz',
    '../data_94/2013/05/03/AIA20130503_0530_0094.npz',
]


# In[ ]:


flare_pre, flare_mid, flare_start, flare_before = np.array([np.load(x)['x'] for x in flare_pre]), np.array([np.load(x)['x'] for x in flare_mid]), np.array([np.load(x)['x'] for x in flare_start]), np.array([np.load(x)['x'] for x in flare_before])


# In[ ]:


flare_pre, flare_mid, flare_start = flare_pre-flare_pre.std(), flare_mid-flare_mid.std(), flare_start-flare_start.std()


# In[ ]:


flare_pre_diff = [flare_pre[i]-flare_pre[i-1] for i in range(1, len(flare_pre))]
flare_mid_diff = [flare_mid[i]-flare_mid[i-1] for i in range(1, len(flare_mid))]
flare_start_diff = [flare_start[i]-flare_start[i-1] for i in range(1, len(flare_start))]

for i in flare_pre_diff:
    i[i<0] = 0
    
for i in flare_mid_diff:
    i[i<0] = 0
    
for i in flare_start_diff:
    i[i<0] = 0


# In[ ]:


flare_pre_peak_diff = flare_pre-flare_pre.std()


# In[ ]:


flare_pre_peak_diff[flare_pre_peak_diff<0]=0


# In[ ]:


plt.figure(figsize=(10, 10))
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

for idx, ax in enumerate(axes.flat):
    ax.imshow(flare_before[idx], cmap='jet')
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")


# In[ ]:


plt.figure(figsize=(10, 10))
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

for idx, ax in enumerate(axes.flat):
    ax.imshow(flare_mid_diff[idx], cmap='jet')
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")


# In[ ]:


plt.figure(figsize=(10, 10))
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

for idx, ax in enumerate(axes.flat):
    ax.imshow(flare_start_diff[idx], cmap='jet')
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")


# In[ ]:




