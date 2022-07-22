from flask import Flask, render_template, Response
from dataprep.data_preparation import DataPreparation
from managers.model_manager import ModelManager
from graphs.graph_manager import GraphManager
from managers.train_manager import TrainManager
from managers.test_manager import TestManager
from model.lstm_model import LSTMModel
import pandas as pd
from utils.custom_callback import CustomCallback


app = Flask(__name__)
@app.route('/')
def main():
    DataPreparation()
    nulls_series = DataPreparation.original_null_list
    bad_cols = DataPreparation.bad_cols
    ranked_features = DataPreparation.ranked_features
    ranked_array = ranked_features.to_numpy()
    num_features_to_include = DataPreparation.num_features_to_include
    feature_names_to_include = ranked_array[0:num_features_to_include, 0]
    return render_template('main.html',nulls_series=nulls_series, bad_cols=bad_cols,
                           ranked_features=ranked_array, features_in_model=feature_names_to_include )

@app.route('/progress-shape-data')
def progress_shape_data():
    DataPreparation.finish_data_prep()
    return Response(DataPreparation.finish_data_prep(), mimetype='text/event-stream')

@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    train_input_shape = (DataPreparation.X_train.shape[1], DataPreparation.X_train.shape[2])
    learning_rate = 0.01
    hidden_layer1_nodes = 128
    hidden_layer2_nodes = 128
    hidden_layer3_nodes = 64
    epochs = 10
    batch_size = 64
    # Create a TrainManager which builds and compiles model.
    train_manager = TrainManager(hidden_layer1_nodes, hidden_layer2_nodes, hidden_layer3_nodes, learning_rate)
    train_history = train_manager.fit_model(DataPreparation.X_train, DataPreparation.y_train, epochs, batch_size)

    encoded_image = GraphManager.plot_history(train_history)
    return encoded_image

@app.route('/test-model', methods=['GET', 'POST'])
def test_model():
    buffer = TestManager.make_test_graph()
    return buffer



if __name__ == '__main__':
    app.run(port=5004, debug=True)