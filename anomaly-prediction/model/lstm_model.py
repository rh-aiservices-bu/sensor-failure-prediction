import keras.models
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow import random
from numpy.random import seed
import tensorflow as tf
from utils.custom_callback import CustomCallback

"""
    Model has 3 hidden layers.
    Input data has 3D's: [samples, timesteps, features]
    :param input_shape tuple of input shape, index 1 and 2, which correspond to number of timesteps and number of features.
    :param n_hidden1_nodes  Number of nodes in hidden layer 1
    :param n_hidden2_nodes  Number of nodes in hidden layer 2
    :param n_hidden3_nodes  Number of nodes in hidden layer 3
    
"""
class LSTMModel():

    def __init__(self, input_shape=None,
                 n_hidden1_nodes=None, n_hidden2_nodes=None, n_hidden3_nodes=None):
        seed_value = 64
        tf.random.set_seed(seed_value)
        self.model = Sequential()
        self.model.add(LSTM(n_hidden1_nodes, input_shape=input_shape, return_sequences=True))
        self.model.add(LeakyReLU(alpha=0.2))

        self.model.add(LSTM(n_hidden2_nodes, return_sequences=True))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.5))

        self.model.add(LSTM(n_hidden3_nodes, return_sequences=True))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(1, activation='sigmoid'))

    def compile(self, learning_rate):
        self.model.compile(loss='binary_crossentropy',
                     optimizer=Adam(learning_rate=learning_rate),
                     metrics=['binary_accuracy'])

    def fit(self, X, y, epochs=None, batch_size=None, callbacks=None, verbose=1, shuffle=True):
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                            verbose=verbose, shuffle=shuffle)
        return history

    def save(self, model_path):
        self.model.save(model_path)

    @staticmethod
    def load_model(model_path):
        return keras.models.load_model(model_path)

    def predict(self, X):
        y = self.model.predict(X)
        return y
