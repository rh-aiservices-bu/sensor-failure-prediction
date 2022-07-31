import joblib

from dataprep.data_source_manager import DataSourceManager
from dataprep.data_preparation import DataPreparation
import pandas as pd


class ProcessRealtimeData:
    def __init__(self, predict_window_size, feature_names, scaler_filename, csv_filename=None):
        self.predict_window_size = predict_window_size
        self.csv_filename = csv_filename
        self.feature_names = feature_names
        # self.feature_vals = []
        self.prediction_buff = []
        self.row_counter = 0
        self.scaler = self.load_scaler(scaler_filename)

    def load_scaler(self, scaler_file_name):
        scaler = joblib.load(scaler_file_name)
        return scaler

    def process_points(self):
        gen = DataSourceManager.csv_line_reader(self.csv_filename)
        while True:
            row = next(gen, None)
            if row is None:
                yield "event: jobfinished\ndata: " + "none" + "\n\n"
                break
            else:
                self.prediction_buff.append(row)
                if self.row_counter >= self.predict_window_size:
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
                    scaled_data = DataPreparation.scale_dataframe(self.scaler, buff_df, self.feature_names)




