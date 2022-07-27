import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def attention_3d_block(hidden_states, series_len=6):
    hidden_size = int(hidden_states.shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
    hidden_states_t = Reshape((hidden_size, series_len), name='attention_input_reshape')(hidden_states_t)
    score_first_part = Dense(series_len, use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    
    return attention_vector

def get_sequence_model():
    inp = Input(shape=(6, 2048))
    x = LSTM(10, return_sequences=True)(inp)
    x = attention_3d_block(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(0.0001))(x)

    rnn_model = Model(inp, output)
    adam_fine = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0002, amsgrad=False)
    rnn_model.compile(
        loss="binary_crossentropy", optimizer=adam_fine, metrics=["accuracy"]
    )
    
    return rnn_model