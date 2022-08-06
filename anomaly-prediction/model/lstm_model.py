import keras.models
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

import tensorflow as tf


class LSTMModel():
    """LSTM model
        Model has 3 hidden layers.
        Input data has 3D's: [samples, timesteps, features]
        :param input_shape tuple of input shape, index 1 and 2, which correspond to number of timesteps and number of features.
        :param n_hidden1_nodes  Number of nodes in hidden layer 1
        :param n_hidden2_nodes  Number of nodes in hidden layer 2
        :param n_hidden3_nodes  Number of nodes in hidden layer 3

    """

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
        """Compile model

        :param learning_rate: Initial learning rate
        :type: float
        :return: none
        """
        self.model.compile(loss='binary_crossentropy',
                     optimizer=Adam(learning_rate=learning_rate),
                     metrics=['binary_accuracy'])

    def fit(self, X, y, epochs=None, batch_size=None, callbacks=None, verbose=1, shuffle=True):
        """Fit Model

        :param X: 3-D numpy array
        :type: ndarray
        :param y: 1-d numpy array
        :type: ndarray
        :param epochs: Number of training epochs
        :type: int
        :param batch_size: Size of each training batch
        :type: int
        :param callbacks: List of callback functions
        :type: list
        :param verbose: Command line verbose flag
        :type: int
        :param shuffle: Whether to shuffle
        :type: boolean
        :return: Fit history
        :type: array
        """
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                            verbose=verbose, shuffle=shuffle)
        return history

    def save(self, model_path):
        """Save model

        :param model_path: Path to saved model
        :type: string
        :return: none
        """
        self.model.save(model_path)


    @staticmethod
    def load_model(model_path):
        """Load model

        :param model_path: Path of saved model
        :type: string
        :return: model from saved file
        """
        return keras.models.load_model(model_path)

    def predict(self, X):
        """Predict

        :param X: x values for LSTM model
        :type  3-D numpy array
        :return: predictions
        :type: array
        """
        y = self.model.predict(X)
        return y
