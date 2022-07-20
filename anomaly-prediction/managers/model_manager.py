from model.lstm_model import LSTMModel
import keras.models

class ModelManager():

    @staticmethod
    def create_model(input_shape, learning_rate, n_hidden1_nodes,
                    n_hidden2_nodes, n_hidden3_nodes=None):
        model = LSTMModel(input_shape, n_hidden1_nodes, n_hidden2_nodes, n_hidden3_nodes)
        model.compile(learning_rate)
        return model

    @staticmethod
    def fit(model, X, y, epochs=None, batch_size=None, callbacks=None):
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1, shuffle=True)
        return history

    @staticmethod
    def load_model(model_path):
        # self.model.load_model(model_path)
        return keras.models.load_model(model_path)

