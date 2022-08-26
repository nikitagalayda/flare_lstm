def SingleConvLSTMModel(batch_size, image_size, sequence_length):
    inp = Input(shape=(sequence_length, image_size, image_size, 1))

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

def ConvLSTMModelAllClass(batch_size, image_size, sequence_length, num_classes=3):
    inp = Input(shape=(sequence_length, image_size, image_size, 1))
    print(inp.shape)
    
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