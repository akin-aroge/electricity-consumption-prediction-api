
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from src import utils
from typing import Tuple
import numpy as np

from sklearn.preprocessing import MinMaxScaler

root_dir = utils.get_proj_root()

class DNN(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1D_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=12,
                          strides=1,
                          activation="relu",
                          padding='causal',
                          input_shape=input_shape)
        self.lstm_1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32, return_sequences=False)
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(8, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.conv1D_1(inputs)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x
    
# TODO: create normalization layer

# class CustomNormalizationLayer(Layer):
#     def __init__(self, mean, std_dev, epsilon=1e-6, **kwargs):
#         super(CustomNormalizationLayer, self).__init__(**kwargs)
#         self.mean = mean
#         self.std_dev = std_dev
#         self.epsilon = epsilon
#         self.trainable = False


#     def call(self, inputs):
#         normalized = (inputs - self.mean) / (self.std_dev + self.epsilon)
#         return normalized

#     def get_config(self):
#         config = super(CustomNormalizationLayer, self).get_config()
#         config.update({'mean': self.mean.tolist(), 'std_dev': self.std_dev.tolist(), 'epsilon': self.epsilon})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         mean = tf.constant(config['mean'])
#         std_dev = tf.constant(config['std_dev'])
#         epsilon = config['epsilon']
#         return cls(mean, std_dev, epsilon)

#     def compute_output_shape(self, input_shape):
#         return input_shape



def create_model(input_shape, learning_rate, optimizer='adam', loss='huber'):

    model = DNN(input_shape=input_shape)

    if optimizer == 'adam':
        optimizer = optimizers.Adam(learning_rate=float(learning_rate))
    elif optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=float(learning_rate))
    elif optimizer == 'sgd':
        optimizer = optimizers.SGD(learning_rate=float(learning_rate))

    if loss == 'huber':
        loss = losses.Huber()
    elif loss == 'mae':
        loss = tf.keras.losses.MAE()

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    return model

def window_dataset(X, y, window_size, batch_size) -> tf.data.Dataset:


    # X = np.concatenate((X, y), axis=1)  # add load data to training

    # N_w (or N_labels) = N-w_s+1
    # so N (X_rows) = N_labels+ws-1
    y = y[window_size-1:]  # label for window starting at X[0] -> y[w_size-1]
    X = X[:len(y)+window_size-1]
    # print('in window', X.shape, y.shape)
        
    data = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=False
    )
    return data

def adapt_scaler(X_train, y_train):
    X_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train.reshape(-1, 1))

    utils.save_value((X_scaler, y_scaler), fname=root_dir.joinpath('models/norm_Xy_scaler.pkl'))

    return X_scaler, y_scaler

def get_scaler():

    X_scaler, y_scaler = utils.load_value(fname=root_dir.joinpath('models/norm_Xy_scaler.pkl'))

    return X_scaler, y_scaler

def get_input_shape(dataset:tf.data.Dataset):

    for X, y in dataset.take(1):
        input_shape = X.shape[1:]

    return input_shape


es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

def train_model( train_data:tuple, lr:float, val_data=None, batch_size=32,  
                window_size=32, n_epochs=10, 
                optimizer='adam', loss='huber',  early_stop=False) -> tuple[DNN, tf.keras.callbacks.History]:
    
    X_train, y_train = train_data
    X_scaler, y_scaler = adapt_scaler(X_train=X_train, y_train=y_train)

    # X_train, y_train = scale_data(X_train, y_train)
    X_train = X_scaler.transform(X_train)
    y_train = y_scaler.transform(y_train.reshape(-1, 1))
    train_data = window_dataset(X_train, y_train, window_size=window_size, batch_size=batch_size)
    # print('in tran_model', train_data.cardinality(), X_train.shape,(X_train.shape[0] - window_size + 1)/batch_size)
    if val_data is not None:
        X_test, y_test = val_data
        X_test = X_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test.reshape(-1, 1))
        val_data = window_dataset(X_test, y_test, window_size=window_size, batch_size=batch_size)

        # create model
    # input_shape = [window_size, X_train.shape[-1]]
    input_shape = get_input_shape(train_data)
    model = create_model(input_shape=input_shape, learning_rate=lr, optimizer=optimizer)

    if early_stop:
        history = model.fit(train_data, validation_data=val_data, 
                            epochs=n_epochs, callbacks=[es_callback])
    elif early_stop is False:
        history = model.fit(train_data, validation_data=val_data, 
                            epochs=n_epochs)  
        
    return model, history





