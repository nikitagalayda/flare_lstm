import os
import numpy as np
import cv2
import tensorflow as tf

class PairAllClassGen(tf.keras.utils.Sequence):
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