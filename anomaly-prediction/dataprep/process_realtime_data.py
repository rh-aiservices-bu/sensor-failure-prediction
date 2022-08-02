import json

import joblib

from dataprep.data_source_manager import DataSourceManager
from dataprep.data_preparation import DataPreparation
import pandas as pd


class ProcessRealtimeData:
    def __init__(self, predict_window_size, feature_names, scaler_filename, pca_filename, csv_filename=None):
        self.predict_window_size = predict_window_size
        self.csv_filename = csv_filename

        self.feature_names = feature_names
        # self.feature_vals = []
        self.prediction_buff = []
        self.row_counter = 0
        self.scaler = self.load_scaler(scaler_filename)
        self.pca = self.load_pca(pca_filename)
        self.stride = 1

    def load_scaler(self, scaler_file_name):
        return joblib.load(scaler_file_name)

    def load_pca(self, pca_filename):
        return joblib.load(pca_filename)

    def process_points(self):
        gen = DataSourceManager.csv_line_reader(self.csv_filename)
        while True:
            row = next(gen, None)
            if row is None:
                yield "event: jobfinished\ndata: " + "none" + "\n\n"
                break
            else:
                row_as_df = pd.DataFrame(row, index=[0])
                row_as_df = row_as_df.set_index('timestamp')
                scaled_data = DataPreparation.scale_dataframe(self.scaler, row_as_df, self.feature_names)
                self.prediction_buff.append(row)
                scaled_buff = None
                if self.row_counter >= 2 * self.predict_window_size:
                    # Keep prediction_buffsize  as predict_window_size
                    self.prediction_buff.pop(0)
                    # convert buffer into df
                    buff_df = pd.DataFrame(self.prediction_buff)
                    # make all feature cols as float
                    cols = buff_df.columns
                    buff_df[cols[1:]] = buff_df[cols[1:]].astype(float)
                    # Add 'alarm' col
                    buff_df['alarm'] = 0
                    # Scale buffer to ndarray.  NOTE:  this is inefficient since n-1 rows have already been scaled
                    scaled_buff = DataPreparation.scale_dataframe(self.scaler, buff_df, self.feature_names)

                    X, y = DataPreparation.make_predict_data(scaled_buff, buff_df['alarm'], self.feature_names,
                                                      self.predict_window_size, self.stride)
                    json_data = self.create_dict(row, y)
                    self.row_counter += 1
                    yield "event: update\ndata: " + json.dumps(json_data) + "\n\n"
                else:
                    self.row_counter += 1
                    scaled_buff = DataPreparation.scale_dataframe(self.scaler, buff_df, self.feature_names)
                    json_data = self.create_dict(scaled_buff, None)
                    yield "event: update\ndata: " + json.dumps(json_data) + "\n\n"
                    # next(gen)

    def create_dict(self, one_row, alarm_arr):
        print()

        plot_dict = {
            'timestamp': one_row['timestamp'],
            'sensor0': one_row['sensor_25'],
            'sensor1': one_row['sensor_11'],
            'sensor2': one_row['sensor_36'],
            'sensor3': one_row['sensor_34'],
            'alarm': [0]
        }
        return plot_dict

