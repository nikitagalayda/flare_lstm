import os
import numpy as np
import cv2
import tensorflow as tf

class CustomDataFeaturesGen(tf.keras.utils.Sequence):
    
    def __init__(self, folder_paths,
                 batch_size,
                 input_size=(6, 2048),
                 shuffle=True):
        
        self.folder_paths = folder_paths.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.folder_paths)
        self.n_category = 2
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.folder_paths)
    
    def __getitem__(self, index):
        batches = self.folder_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)  
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_input(self, folder):
        features = []
        for subdir, dirs, files in os.walk(folder):
            for f in files:
                features.append(os.path.join(subdir, f.decode(encoding='UTF-8')))
        features = sorted(features)
        # for ft in features:
        #     print(ft)
        features = [np.load(x) for x in features]
        features = [x.reshape(x.shape[3]) for x in features]
        # images = [preprocessing.normalize(np.load(x)) for x in images]
        if len(features) != 6:
            print(folder)
        features = np.array(features)
        
        return features

    def __get_output(self, path, num_classes=2):
        label = None
        folder = path.rsplit('/')[-2]
        if folder == 'positive':
            label = 1
        elif folder == 'negative':
            label = 0
        
        return label
        # return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(x) for x in batches])
        y_batch = np.asarray([self.__get_output(y, self.n_category) for y in batches])

        return X_batch, y_batch