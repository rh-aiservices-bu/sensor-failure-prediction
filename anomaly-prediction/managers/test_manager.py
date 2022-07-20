from managers.model_manager import ModelManager
from managers.train_manager import TrainManager
from dataprep.data_preparation import DataPreparation
from graphs.graph_manager import GraphManager
import pandas as pd
import tensorflow as tf
import numpy as np

class TestManager:

    @staticmethod
    def get_model():
        model = TrainManager.model
        return model
    @staticmethod
    def get_data():
        pass

    @staticmethod
    def get_failure_time():
        failure_time_test = DataPreparation.df_test_pca[DataPreparation.df_test_pca['machine_status'] == 1].index[0]
        return failure_time_test
    @staticmethod
    def make_test_graph():
        failure_time = TestManager.get_failure_time()
        start_time_offset = DataPreparation.test_time_window_dimensions[0]
        end_time_offset = DataPreparation.test_time_window_dimensions[1]
        # test_time_window_dimensions[0] is the start time offset from failure time (min)
        # test_time_window_dimensions[1] is the end time offset from failure time (min)
        windows_start = failure_time - pd.Timedelta(seconds=60 * start_time_offset)
        windows_end = failure_time - pd.Timedelta(seconds=60 * end_time_offset)
        df_test_window = DataPreparation.df_test_pca.loc[windows_start:windows_end, :]

        y_test_predictions = TrainManager.model.predict(DataPreparation.X_test).flatten()

        y_test_predictions = tf.round(y_test_predictions)
        df_test_window['alarm'] = np.append(np.zeros(DataPreparation.window_size), (y_test_predictions))

        col_list = list(DataPreparation.df_test_pca.columns)
        col_features = col_list[:-2]  # only features
        alarm_times = df_test_window[df_test_window['alarm'] == 1].index.to_list()

        """
        # vertical lines
        for xc in df_test_window[df_test_window['alarm'] == 1].index.to_list():
            plt.axvline(x=xc, c='red')

        df_test['machine_status'].plot()
        plt.xlim([windows_start, windows_end + pd.Timedelta(seconds=60 * 200)])
        """

        buffer = GraphManager.plot_test(df_test_window[col_features], alarm_times)
        return buffer



