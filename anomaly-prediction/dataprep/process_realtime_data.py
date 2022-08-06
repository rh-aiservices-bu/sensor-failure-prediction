import json
import joblib
from dataprep.data_source_manager import DataSourceManager
from dataprep.data_preparation import DataPreparation
from model.lstm_model import LSTMModel
import pandas as pd


class ProcessRealtimeData:
    """Class used to process, predict and yield plotting data for one point

    """
    def __init__(self, predict_window_size, scaler_filename, pca_filename,
                 means_filename, bad_cols_filename, model_filename, csv_filename=None):
        """Class initializer (Constructor)

        Just so that the model does not have to be retrained each time we want to make a prediction,
        the values of scaler, pca, column means, and model that are all calculated at train time are saved.
        So this constructor retrieves all those values necessary to prepare each point for prediction.
        :param predict_window_size: Size in time points of the prediction window.  For example, a value of 20
        means that the previous 19 points will be used to predict the 20th point
        :type: int
        :param scaler_filename: File name of the saved scaler
        :type: string
        :param pca_filename: File name of the saved PCA transformation data
        :type: string
        :param means_filename: File name of the saved column means of the training data.  The column means are used
        to replace any nan's in the prediction data
        :param bad_cols_filename: File name of column names that have been excluded from training.
        :type: string
        :param model_filename: File name of the saved trained model
        :type: string
        :param csv_filename: File name of the optional csv file used as a data source
        :type: string
        """
        self.predict_window_size = predict_window_size
        self.csv_filename = csv_filename
        self.prediction_buff = []  # This buffer will be a list of DataFrames, where each row of predict data is a
                                    # Dataframe
        self.row_counter = 0
        self.scaler = self.load_scaler(scaler_filename)
        self.pca = self.load_pca(pca_filename)
        self.means = self.load_means(means_filename)
        self.bad_cols = self.load_bad_cols(bad_cols_filename)
        self.model = self.load_model(model_filename)

        self.stride = 1

    def load_scaler(self, scaler_file_name):
        return joblib.load(scaler_file_name)

    def load_pca(self, pca_filename):
        return joblib.load(pca_filename)

    def load_bad_cols(self, bad_cols_filename):
        return joblib.load(bad_cols_filename)

    def load_means(self, means_filename):
        return joblib.load(means_filename)

    def load_model(self, model_filename):
        #return joblib.load(model_filename)
        return LSTMModel.load_model('static/trained_model')

    def process_points(self):
        """Process one point from the prediction data
        A data source generator is used to retrieve prediction data one point at a time.
        :return: none  NOTE:  This class method is a generator, so there is no return. However it does yield
        a JSON serialized dictionary that contains the data for plotting the prediction graph
        """

        # gen is a generator that is an iterable of dictionaries. Each dictionary contains one row of prediction data
        # including timestamp and PC data
        gen = DataSourceManager.csv_line_reader(self.csv_filename)
        while True:
            row = next(gen, None)  # Get next row where row is a dictionary
            if row is None:
                # The value of this yield, when received by the client javascript, will shut down the socket that is
                # used for pushing the prediction data.
                yield "event: jobfinished\ndata: " + "none" + "\n\n"
                break  # Terminate this event loop
            else:
                # Convert row dictionary to DataFrame
                row_as_df = pd.DataFrame(row, index=[0])
                # Set the timestamp values of the prediction window's start and end
                predict_window_end = \
                    pd.to_datetime(row_as_df['timestamp'][0])  # Newest time
                predict_window_start = \
                    pd.to_datetime(predict_window_end - pd.Timedelta(seconds=60*(2 * self.predict_window_size - 1))) # Start 20 min before end

                # df has index of timestamp
                row_as_df.set_index('timestamp', inplace=True)
                # Drop bad columns
                DataPreparation.drop_bad_cols(row_as_df, self.bad_cols)
                # Replace any nan with mean for that column that was obtained when model was trained
                row_as_df.fillna(value=self.means, inplace=True)
                # Drop col 'machine_status'
                row_as_df.drop('machine_status', axis=1, inplace=True)

                # Transform with self.scaler.
                # scaled_data is a numpy array
                scaled_data = self.scaler.transform(row_as_df)
                # Number of featured (PC's) that were determined when the training data was transformed by the PCA
                num_features = self.pca.n_components_
                # Add 'alarm' col
                row_as_df['alarm'] = 0
                # Add 'machine_status' temporarily since DataPreparation.transform_df_by_pca expects that column
                row_as_df['machine_status'] = 0
                # Transform with self.pca.  The resulting df has the same index as the row_as_df.
                df_row_transformed = DataPreparation.transform_df_by_pca(self.pca, row_as_df,
                                                                      scaled_data, num_features)
                row_as_df.drop('machine_status', axis=1, inplace=True)
                # TODO:  Currently plotting using PC's.  If we want to plot raw sensor data, separate raw sensor data
                #  in row to be used for plotting.
                # Use the pca_transformed data for prediction.
                feature_names = df_row_transformed.columns[:-2]
                # Make buff a df of scaled, PCA transformed data plus 'alarm' column uninitialized
                self.prediction_buff.append(df_row_transformed)
                # If the prediction_buff is not full, do not attempt to do a predict on the new data point
                # Buffer size will be 2 * the predict window size
                if self.row_counter >= 2 * self.predict_window_size:
                    # Keep prediction_buffsize  as 2  * predict_window_size by popping oldest point from buffer
                    self.prediction_buff.pop(0)
                    # convert buffer into df by concat list of df's in buffer
                    buff_df = pd.concat(self.prediction_buff) # has timestamp index and 'alarm' col
                    # Convert buff_df index to datetime so the index will be compatible with loc[]
                    buff_df.index = pd.to_datetime(buff_df.index)
                    df_predict_window = buff_df.loc[predict_window_start:predict_window_end]

                    # Prepare the data in the buffer so that it can be used in the LSTM model for prediction
                    X, y = DataPreparation.make_predict_data(df_predict_window, feature_names,
                                                      self.predict_window_size)
                    # Get prediction as array with length = self.predict_window_size
                    y_predict = self.model.predict(X).flatten()
                    # Get the last element of the y_predict list for plotting.  Last element in y_predict
                    # is the most current point which makes it the prediction point.
                    json_data = self.__create_dict(df_predict_window, True, alarm_value=y_predict[19])
                    self.row_counter += 1
                    yield "event: update\ndata: " + json.dumps(json_data) + "\n\n"
                else: # buffer not yet filled, just plot sensor data with no prediction
                    self.row_counter += 1
                    json_data = self.__create_dict(df_row_transformed, False)
                    yield "event: update\ndata: " + json.dumps(json_data) + "\n\n"

    def __create_dict(self, one_row_df, alarm, alarm_value=None):
        """Private method to create a dictionary

        :param one_row_df: One row as a DataFrame
        :type: DataFrame
        :param alarm: A Boolean that determines whether the data in one_row_df is to be used for prediction. If True,
        the data in one_row_df will contain a prediction, False if the data will just be plotted without any prediction.
        The False case arises when the buffer is not yet full and we do not yet have enough data to make a prediction.
        :type: boolean
        :param alarm_value: Value of the prediction
        :type: float
        :return: A dictionary of data that will be used for plotting the real time prediction
        """
        if alarm:
            alarm_val = alarm_value
        else:
            alarm_val = None

        plot_dict = {


            'timestamp': str(one_row_df.index[0]),
            'pc1': one_row_df['pc1'].values[0],
            'pc2': one_row_df['pc2'].values[0],
            'alarm': str(alarm_val)

        }
        return plot_dict

