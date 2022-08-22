class InitialDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, folder_paths,
                 batch_size,
                 input_size=(20, 64, 64),
                 shuffle=True):
        
        self.folder_paths = folder_paths.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.folder_paths)
        self.n_category = 2
        # self.n_name = df[y_col['name']].nunique()
        # self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.folder_paths)
    
    def __getitem__(self, index):
        batches = self.folder_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)  
        X = np.expand_dims(X, axis=4)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_input(self, folder):
        images = []
        for subdir, dirs, files in os.walk(folder):
            for f in files:
                images.append(os.path.join(subdir, f.decode(encoding='UTF-8')))
        images = sorted(images)
        # for im in images:
        #     print(im)
        images = [np.load(x) for x in images]
        # images = [NormalizeData(x) for x in images]
        if len(images) != 20:
            print(folder)
        # images = [cv2.resize(x, (128, 128), interpolation = cv2.INTER_AREA) for x in images]
        images  = np.array(images)
        return images

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
        # Generates data containing batch_size samples
        # path_batch = batches[self.X_col['path']]
        # category_batch = batches[self.y_col['type']]
        # print(batches)
        X_batch = np.asarray([self.__get_input(x) for x in batches])
        y_batch = np.asarray([self.__get_output(y, self.n_category) for y in batches])

        return X_batch, y_batch
    
