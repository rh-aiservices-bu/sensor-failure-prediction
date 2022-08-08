
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from utils.data_type_enum import DataTypeEnum
import joblib
import tensorflow as tf

from services.sensor_data_service import SensorDataServiceCSV


class DataPreparation:
    """Prepare training and testing data
    This class holds in class variables all the prepared training and testing data.  In addition there is a method called
    make_predict_data() that transforms the prepared data into a form that is recognized by the LSTM model
    """
    failure_times = None
    pca = None
    # 3 DataFrames of scaled and PCA data
    df_train_pca = None
    df_test_pca = None
    df_val_pca = None
    # ndarrays for train, test, val
    X_train = None
    X_val = None
    X_test = None
    y_train = None
    y_val = None
    y_test = None
    train_time_window_dimensions = [96 * 60, 12 * 60, 12 * 60, 5]  # 96h, 12h, 12h, 5min

    test_time_window_dimensions = [70 * 60, 5]  # 70h 5 min
    stride = 5
    window_size = 20
    whole_dataframe = None
    original_null_list = None
    ranked_features = None
    num_features_to_include = None
    bad_cols = ['Unnamed: 0', 'sensor_00', 'sensor_15', 'sensor_50', 'sensor_51']
    job_size = 0
    progress_counter = 0
    min_max_scaler = None
    pca = None

    def __init__(self):
        DataPreparation.do_automated_data_prep()
        joblib.dump(DataPreparation.bad_cols, 'static/bad_cols.gz')

    @staticmethod
    def get_df():
        """Get all sensor data as DataFrame

        :return: Sensor as a dataframe
        :type: Pandas DataFrame
        """
        df = SensorDataServiceCSV.get_all_sensor_data()
        return df

    # Drop columns in given list
    @staticmethod
    def drop_bad_cols(df, col_list):
        """Drop unusable columns inplace

        :param df: DataFrame
        :type: Pandas DataFrame
        :param col_list: List of unusable column names
        :return: none
        """
        df.drop(col_list, axis=1, inplace=True)

    # How many nulls in each col.
    # return Series with Index (col names) and values (number of nulls for col)
    @staticmethod
    def get_null_list(df):
        """Get list of sums of nulls.
        Get list by columns showing how many nulls are in each column
        :param df: DataFrame
        :type: Pandas DataFrame
        :return: List of sums of nulls
        :type: List
        """
        nulls_series = df.isnull().sum()
        DataPreparation.original_null_list = nulls_series
        return nulls_series

    @staticmethod
    def replace_nan_with_mean(df):
        """
        Replace NaN columnwise with mean of each column inplace
        :param df: DataFrame
        :type: Pandas DataFrame
        :return: none
        """
        df_cols = df.columns
        df.fillna(value=df[df_cols].mean(), inplace=True)

    @staticmethod
    def machine_status_to_numeric(df):
        """Make 'machine_status" column numeric
        Numeric values are 0: 'NORMAL';, 1: 'BROKEN', 0.5: 'RECOVERING'
        :param df: DataFrame
        :type: Pandas DataFrame
        :return: none
        """
        status_values = [(df['machine_status'] == 'NORMAL'), (df['machine_status'] == 'BROKEN'),
                         (df['machine_status'] == 'RECOVERING')]
        numeric_status_values = [0, 1, 0.5]

        df['machine_status'] = np.select(status_values, numeric_status_values, default=0)

    @staticmethod
    def add_target_col(df, failure_times, start_offset, end_offset):
        """Create a new column that will contain values that indicate time window to do prediction
        :param df the original dataframe
        :type: Pandas DataFrame
        :param start_offset  Starting offset from a failure time (in minutes)
        :type: int
        :param end_offset    Ending offset from a failure time (in minutes)
        :type: int
        :param df: Dataframe
        :type: Pandas DataFrame
        :param failure_times: List of failure times ( where 'machine_status' has a value of 1
        :type: List of string timestamps

        :return: none
        """
        df['alarm'] = df['machine_status']
        for i, failure_time in enumerate(failure_times):
            start_predic_time = failure_time - pd.Timedelta(
                seconds=60 * start_offset)  # mins before the failure time
            stop_predic_time = failure_time - pd.Timedelta(
                seconds=60 * end_offset)  # mins before the failure time
            df.loc[start_predic_time:stop_predic_time,
            'alarm'] = 2  # can not use 1, because 1 indicates the machine failure time

    @staticmethod
    def get_failure_times(df):
        """Get DataFrame of rows where 'machine_status' has a value of 1

        :param df: DataFrame
        :return: Failure times
        :type: DatetimeIndex
        """
        return df[df['machine_status'] == 1].index

    '''
      - the data before and two hours past the first failure is used as validation dataset
      - the data two hours after the first failure and two hours after the second failure is used as test dataset
      - the data two hours after the second failure is used as training dataset
      '''

    @staticmethod
    def separate_data(df, failure_times):
        """Slice data by failure times
        If there are 7 failure times, then produce 7 slices as DataFrames

        :param df: All Data
        :type: Pandas DataFrame
        :param failure_times: DatetimeIndex
        :return: DataFrames for training, validation and testing
        :type: tuple of DataFrames
        """
        df_val = df.loc[:(failure_times[0] + pd.Timedelta(seconds=60 * 120)), :]
        df_test = df.loc[(failure_times[0] + pd.Timedelta(seconds=60 * 120)):(
                failure_times[1] + pd.Timedelta(seconds=60 * 120)), :]
        df_train = df.loc[failure_times[1] + pd.Timedelta(seconds=60 * 120):, :]

        return df_train, df_val, df_test

    @staticmethod
    def get_scaler(training_df):
        """Get a scaler for training data
        Apply MinManScaler to the training data using the range (0, 1)
        :param training_df:
        :type: Pandas DataFrame
        :param sensor_names: List of sensor names(feature names)
        :type: list
        :return: scaler for the training data
        :type: MinMaxScaler
        """
        sensor_names = training_df.columns[:-2]
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = min_max_scaler.fit(training_df[sensor_names])

        return scaler

    @staticmethod
    def scale_dataframe(scaler, data_df):
        """Transform given dataframe using given scaler as applied to the given columns

        :param scaler: The scaler that has been fit to the dataframe
        :type: MinMaxScaler
        :param data_df: Dataframe to be scaled
        :type: Pandas DataFrame
        :param sensor_names: List of sensor names
        :type: List of string
        :return: array of scaled data
        :type: ndarray
        """
        sensor_names = data_df.columns[:-2]
        scaled_data = scaler.transform(data_df[sensor_names])
        return scaled_data

    @staticmethod
    def get_PCA(train_scaled_data, num_components):
        """Get PCA that has been fit to the scaled training data
        PCA is the Principal Component Analysis.  Training data features are each ranked by their maximum variance from
        all the other features.
        :param train_scaled_data: scaled training data
        :type: ndarray
        :param num_components: Number of features in the training data
        :return: Fit PCA
        :type: PCA
        """
        pca = PCA(n_components=num_components).fit(train_scaled_data)
        return pca

    @staticmethod
    def get_ranked_features(fit_pca, feature_names):
        """Get a DataFrame of features ranked by their Variance Ratio.
        This method finds the PCA rankings of the given features.  The param, feature_names are used to map
        the PCA rankings to the actual feature names, since the PCA.transform() method only returns PCA feature names of
        p0, p1, p2, etc. and not the actual sensor names.  The resulting DataFrame will have two columns: 'Ranked Features',
        and 'Variance Ratio'.  The feature with the highest Variance Ratio is on top.  This ranking allows the user
        to determine which features exhibit the largest Variance Ratios.
        :param fit_pca: The PCA that has been fit on the scaled training data
        :type: PCA
        :param feature_names: List of feature name
        :return: DataFrame of ranked features
        :type: Pandas DataFrame
        """
        feature_names = ['pc' + str(i + 1) for i in range(20)]

        num_components = fit_pca.n_components_

        # bundle ranked important feature names with variance ratio
        name_dict = {feature_names[i]: fit_pca.explained_variance_ratio_[i] for i in range(num_components)}

        df_ranked_features = pd.DataFrame(name_dict.items())
        df_ranked_features.columns = ['Ranked Components', 'Variance Ratio']
        return df_ranked_features

    @staticmethod
    def transform_df_by_pca(pca, df_data, scaled_data, num_features_to_include):
        """Reduce the number of features in the training data by the parameter num_features_to_include.

        :param pca: The PCA for the training data
        :type: PCA
        :param df_data: Training data dataframe
        :type: Pandas DataFrame
        :param scaled_data: Array of scaled data
        :type: ndarray
        :param num_features_to_include: Number of features to include.  Currently, this number is auto-determined
        by the num of PC's that were chosen by the PCA to reach 95% ( sum of explained_variance_ratio)
        :type: int
        :param sensor_names: List of sensor names that will be included
        :type: List of string
        :return: DataFrame with data that has been scaled and whose dimensions have been reduced.  This DataFrame has
        the same index as the param df_data
        :type: Pandas DataFrame
        """

        data_transformed = pca.transform(scaled_data)  # ndarray
        df_transformed = pd.DataFrame(data_transformed)
        pcs = ['pc' + str(i + 1) for i in range(pca.n_components_)]
        df_transformed.columns = pcs

        df_transformed.index = df_data.index
        df_smaller = df_transformed[pcs[:num_features_to_include]]
        df_smaller['machine_status'] = df_data['machine_status'].values
        df_smaller['alarm'] = df_data['alarm'].values

        return df_smaller

    """
    Generate LSTM data for time series segment used for training, testing and predicting

    @:param df  The train, or test, or validation df
    @:param failure_times A Datetimeindex object that contains the failure times found in the df
        NOTE:  A failure is described in the time series by one time point, where 'machine_status' is 1.
    @:param timewindow_dimensions A Series of 4 values that are each offsets of time measured from a 
        failure time.  So if the Series is (A,B,C,D), then A represents the offset from the failure time
        that defines the start time of the time series. B represents the offset from the failure time where
        the time series will end.  C 
    @:param window_len number of time increments that will make one data sample
    @:param stride
     @:param data_type is one of the values in DataTypeEnum
    """

    @staticmethod
    def timeseries_before_failure(df, failure_times, feature_names,
                                  timewindow_dimensions, window_len, stride, data_type):
        """Generate data samples that have a shape that is compatible with the LSTM layers
        Generate data samples using the time windows ahead of each machine failure time;
        Transform original df from a 2D array [samples, features] to a 3D array [samples, timesteps, features*window_len]
        :param df: Dataframe of scaled, PCA-transformed data
        :type: Pandas Dataframe
        :param failure_times: List of failure times as determined by the condition that 'machine_status' is 1 (BROKEN)
        :type: list of string dates
        :param feature_names: List of PC col names ('PC1', 'PC2', ..... etc)
        :type: list of string
        :param timewindow_dimensions: List of time (in minutes) offsets from the failure time for start and end
        of the window under consideration
        :type: list
        :param window_len: How many time points will be used in prediction.  Namely, how many points will be used to
        predict next time point.
        :type: int
        :param stride: How many points will be batched for train or test within window_len time points
        :type: int
        :param data_type: One of DataTypeEnum ( TRAIN = 1, TEST = 2, VAL = 3)
        NOTE:  There is no return value, however the the 3-D ndarrays are stored in class variables of this
        class (DataPreparation)
        :return: none
        """

        if data_type is not DataTypeEnum.TRAIN:
            stride = 1
        X = np.empty((1, 1, window_len * len(feature_names)),
                     float)  # [samples, timesteps, features].  Samples will be appended below
        Y = np.empty((1), float)

        # For each failure_time, generate two arrays of data. First array starts well (timewindow_for_use[0]) before failure time
        # and ends short(timewindow_for_use[1]) before the failure.
        # Second time array starts where first array ends(timewindow_for_use[2] and ends close to the failure (timewindow_for_use[3])
        for dummy, failure_time in enumerate(failure_times):
            windows_start = failure_time - pd.Timedelta(
                seconds=60 * timewindow_dimensions[0])  # mins before the failure time
            windows_end = failure_time - pd.Timedelta(
                seconds=60 * timewindow_dimensions[1])  # mins before the failure time

            # Feature data
            df_prefailure_single_window_feature = df.loc[windows_start:windows_end, feature_names]
            # Label data
            df_prefailure_single_window_target = df.loc[windows_start:windows_end, 'alarm']

            # Convert feature df and label df to lists
            data_aslist = df_prefailure_single_window_feature.to_numpy().tolist()
            targets_aslist = df_prefailure_single_window_target.tolist()

            data_gen1 = TimeseriesGenerator(data_aslist, targets_aslist, window_len,
                                            stride=stride,
                                            sampling_rate=1, batch_size=1, shuffle=(data_type == DataTypeEnum.TRAIN))
            len_gen1 = len(data_gen1)

            print("len_gen1: {}   Train boolean: {}".format(len_gen1, data_type))

            for i in range(len(data_gen1)):
                x, y = data_gen1[i]
                x = np.transpose(x).flatten()
                x = x.reshape((1, 1, len(x)))
                X = np.append(X, x, axis=0)
                Y = np.append(Y, y / 2,
                              axis=0)  # alarm windows are marked as 2, however, for the model,  use 1 becasue of the sigmoid function.
                DataPreparation.progress_counter += 1
                yield "event: inprogress\ndata: " + str(DataPreparation.progress_counter) + "\n\n"
            if data_type == DataTypeEnum.TRAIN:
                # for alarm window, num stride=1
                windows_start = failure_time - pd.Timedelta(
                    seconds=60 * timewindow_dimensions[2])  # mins before the failure time
                windows_end = failure_time - pd.Timedelta(
                    seconds=60 * timewindow_dimensions[3])  # mins before the failure time

                time_delt_min = windows_end - windows_start

                df_prefailure_single_window_feature = df.loc[windows_start:windows_end, feature_names]
                df_prefailure_single_window_target = df.loc[windows_start:windows_end, 'alarm']

                # Convert feature data into a list of groups of 4 (len(feature_cols))
                data = df_prefailure_single_window_feature.to_numpy().tolist()
                targets = df_prefailure_single_window_target.tolist()

                data_gen2 = TimeseriesGenerator(data, targets, window_len, stride=1,
                                                sampling_rate=1, batch_size=1, shuffle=True)
                len_gen2 = len(data_gen2)

                print("len_gen2: {}    Train boolean: {}".format(len_gen2, data_type))
                for i in range(len(data_gen2)):
                    x, y = data_gen2[i]
                    x = np.transpose(x).flatten()
                    x = x.reshape((1, 1, len(x)))
                    X = np.append(X, x, axis=0)
                    Y = np.append(Y, y / 2,
                                  axis=0)  # alarm windows are marked as 2, however, for the model,  use 1 becasue of the sigmoid function.
                    DataPreparation.progress_counter += 1
                    yield "event: inprogress\ndata: " + str(DataPreparation.progress_counter) + "\n\n"
        # remove samples where y is neither 0 nor 1
        id_keep = [i for i, y in enumerate(Y) if (y == 1) | (y == 0)]
        y_data = Y[id_keep]
        X_data = X[id_keep][:, :]
        print("Actual counter: {}    Train boolean: {}".format(DataPreparation.progress_counter, data_type))
        if data_type == DataTypeEnum.TRAIN:
            DataPreparation.X_train = X_data
            DataPreparation.y_train = y_data
        elif data_type == DataTypeEnum.TEST:
            DataPreparation.X_test = X_data
            DataPreparation.y_test = y_data
        else:
            DataPreparation.X_val = X_data
            DataPreparation.y_val = y_data

    @staticmethod
    def make_predict_data(pca_df, feature_names,
                          window_len):
        """Transform 2D data to 3D that is compatible with LSTM layers.
         Generate data samples using the time windows ahead of each machine failure time;
        window_len: how many data points from each feature will be used to make one sample for the model.
        stride: sliding window size
        Transform original df from a 2D array [samples, features] to a 3D array [samples, timesteps, features*window_len]
        :param pca_df: Dataframe of the prediction data that has been scaled and PCA transformed
        :type: DataFrame
        :param feature_names: List of PCA feature names (PC1, PC2, etc.)
        :type: list
        :param window_len:  Length of the prediction window in terms of data points.  This number determines how many
        data points will be used to predict the next point.  For example, if length is 20, it means that 19 points will be
        used to predict the 20th point

        :return: Feature data and label data in a form that can be used by the LSTM model
        :type: tuple.  First entry is a numpy array of shape (samples_size, 1, window_len * len(feature_names), second
        entry is a numpy array of predictions

        """

        stride = 1
        X = np.empty((1, 1, window_len * len(feature_names)),
                     float)  # [samples, 1, n_features*window_len].  Samples will be appended below
        Y = np.empty((1), float)

        # Convert feature df and label df to lists
        # using pca_df get data for feature cols and targets for 'alarm' col
        data_as_list = pca_df[feature_names].to_numpy().tolist()
        targets_as_list = pca_df['alarm'].tolist()

        data_gen1 = TimeseriesGenerator(data_as_list, targets_as_list, window_len,
                                        stride=stride,
                                        sampling_rate=1, batch_size=1, shuffle=False)

        for i in range(len(data_gen1)):
            x, y = data_gen1[i]
            x = np.transpose(x).flatten()
            x = x.reshape((1, 1, len(x)))
            X = np.append(X, x, axis=0)
            Y = np.append(Y, y / 2,
                          axis=0)  # alarm windows are marked as 2, however, for the model,  use 1 becasue of the sigmoid function.
            # DataPreparation.progress_counter += 1
            # yield "event: inprogress\ndata: " + str(DataPreparation.progress_counter) + "\n\n"

        # remove samples where y is neither 0 nor 1
        id_keep = [i for i, y in enumerate(Y) if (tf.round(y) == 1) or (tf.round(y) == 0)]
        # First row is zeros, so delete it
        id_keep = np.delete(id_keep, 0)
        y_data = Y[id_keep]
        X_data = X[id_keep][:, :]
        return X_data, y_data


    # Prepare all data to be used for train and testing.  Store the results in Class Variables for easy retrieval
    @staticmethod
    def do_automated_data_prep():
        """Initial Data preparation
        Data preparation is divided into 2 parts.  This method is part 1.  In this method, train and test data
        have nan's replaced  by the means of each column.  The data is scaled and then transformed by PCA.  In addition,
        the scaler and PCA transformation matrix are saved to the static project folder.  This method is done as an
        asynchronous call so that the web page can load and not be blocked until this method finishes.
        :return: none. NOTE: even though this method has no return values, it still puts all of the prepared data into
        class variables of this class.
        """
        df = DataPreparation.get_df()
        # Get a series that shows how many nulls in each column.
        count_nulls_per_column = DataPreparation.get_null_list(df)
        # After human intervention, determine which cols to drop based on the nulls per column above.
        # Eventually the GUI will present the nuls count per column and the user will select cols to drop
        bad_cols = ['Unnamed: 0', 'sensor_00', 'sensor_15', 'sensor_50', 'sensor_51']
        DataPreparation.drop_bad_cols(df, bad_cols)

        # Change values of col 'machine_status' to numeric values
        DataPreparation.machine_status_to_numeric(df)

        failure_times = DataPreparation.get_failure_times(df)
        # Specify start and stop time offsets in minutes for target col.
        start_time_offset = 12 * 60  # 12 hrs
        stop_time_offset = 1  # 1 min
        # Add a target col
        DataPreparation.add_target_col(df, failure_times, start_time_offset, stop_time_offset)
        # Select train, validation and test data based on failure times
        df_train, df_val, df_test = DataPreparation.separate_data(df, failure_times)

        # Replace all NaN with mean in each column
        DataPreparation.replace_nan_with_mean(df_train)
        DataPreparation.replace_nan_with_mean(df_val)
        DataPreparation.replace_nan_with_mean(df_test)

        # Calculate mean of each column in the training df.  Then save the mean df.
        mean_df = df_train.mean()
        joblib.dump(mean_df, 'static/mean.gz')

        #  Get scaler that has been fit to training data
        DataPreparation.min_max_scaler = DataPreparation.get_scaler(df_train)
        # Save scaler for use in prediction
        joblib.dump(DataPreparation.min_max_scaler, 'static/training_scaler.gz')

        # Scale (transform) using the previously formed min_max_scaler.  Results are ndarray
        # These scaled arrays are for all the sensors since PCA needs scaled data
        DataPreparation.scaled_train = DataPreparation.scale_dataframe(DataPreparation.min_max_scaler, df_train)
        DataPreparation.scaled_test = DataPreparation.scale_dataframe(DataPreparation.min_max_scaler, df_test)
        DataPreparation.scaled_val = DataPreparation.scale_dataframe(DataPreparation.min_max_scaler, df_val)

        # This value will give the number of PC to make the sum of explained_variance_ratio reach 0.95
        num_top_components = 0.95
        # Get PCA df to determine which top features to include
        pca_fit = DataPreparation.get_PCA(DataPreparation.scaled_train, num_top_components)
        DataPreparation.pca = pca_fit
        # Get the number of PC that are needed to make the sum of explained_variance_ratio reach 0.95
        num_features_to_include = pca_fit.n_components_
        # Save PCA for use on prediction data
        joblib.dump(DataPreparation.pca, 'static/pca.gz')
        # Get Dataframe of ranked features determined by PCA
        DataPreparation.ranked_features = DataPreparation.get_ranked_features(DataPreparation.pca, df_train.columns)

        DataPreparation.num_features_to_include = num_features_to_include
        DataPreparation.df_train_pca = DataPreparation.transform_df_by_pca(
            DataPreparation.pca, df_train, DataPreparation.scaled_train,
            num_features_to_include)
        DataPreparation.df_test_pca = DataPreparation.transform_df_by_pca(
            DataPreparation.pca, df_test, DataPreparation.scaled_test,
            num_features_to_include)
        DataPreparation.df_val_pca = DataPreparation.transform_df_by_pca(
            DataPreparation.pca, df_val, DataPreparation.scaled_val,
            num_features_to_include)


    # Finish data prep that wasn't done in do_automated_data_prep().
    # This method should be done as an asychronous process so that the page does not get blocked.
    @staticmethod
    def finish_data_prep():
        """Finish part 2 of the data preparation.  The processing done in this method is time intensive, so it is done
        as a generator that yields progress values back to the browser.

        :return: none.  NOTE: this method is a generator and yields progress values
        """

        feature_names = DataPreparation.df_train_pca.columns.tolist()[:-2]

        train_failure_times = DataPreparation.get_failure_times(DataPreparation.df_train_pca)
        test_failure_times = DataPreparation.get_failure_times(DataPreparation.df_test_pca)
        val_failure_times = DataPreparation.get_failure_times(DataPreparation.df_val_pca)

        DataPreparation.job_size = DataPreparation.__calculate_job_size(train_failure_times, test_failure_times,
                                                                      val_failure_times)

        print("Calculated job size:  {}".format(DataPreparation.job_size))

        DataPreparation.progress_counter = 0
        yield "event: initialize\ndata: " + str(DataPreparation.job_size) + "\n\n"

        # Train windows use DataPreparation.stride for first windows, then stride=1 for second group.

        yield from DataPreparation.timeseries_before_failure(DataPreparation.df_train_pca, train_failure_times,
                                                             feature_names,
                                                             DataPreparation.train_time_window_dimensions,
                                                             DataPreparation.window_size, DataPreparation.stride,
                                                             data_type=DataTypeEnum.TRAIN)

        # Test and Val use stride = 1 by default, no matter what gets passed in the call.

        yield from DataPreparation.timeseries_before_failure(DataPreparation.df_test_pca, test_failure_times,
                                                             feature_names,
                                                             DataPreparation.test_time_window_dimensions,
                                                             DataPreparation.window_size, DataPreparation.stride,
                                                             data_type=DataTypeEnum.TEST)

        # Use same window dimensions for test and val

        yield from DataPreparation.timeseries_before_failure(DataPreparation.df_val_pca, val_failure_times,
                                                             feature_names,
                                                             DataPreparation.test_time_window_dimensions,
                                                             DataPreparation.window_size, DataPreparation.stride,
                                                             data_type=DataTypeEnum.VAL)

        yield "event: jobfinished\ndata: " + "\n\n"


    @staticmethod
    def __calculate_job_size(train_failure_times, test_failure_times, val_failure_times):
        """Calculate job size for use in progress bar.

        :param train_failure_times: Failure times found in training data
        :type: list of dates
        :param test_failure_times: Failure times found in test data
        :type: list of dates
        :param val_failure_times: Failure times found in validation data
        :return: job size
        :type: int
        """
        progress_adjustment_gen1 = -3
        progress_adjustment_gen2 = -19
        job_size = 0
        # Training iterations
        for failure_time in train_failure_times:
            first_window_size_in_minutes = \
                DataPreparation.calculate_time_window_delta(failure_time,
                                                            DataPreparation.train_time_window_dimensions[0],
                                                            DataPreparation.train_time_window_dimensions[1]) / 5
            job_size += first_window_size_in_minutes + progress_adjustment_gen1
            second_window_size_in_minutes = \
                DataPreparation.calculate_time_window_delta(failure_time,
                                                            DataPreparation.train_time_window_dimensions[2],
                                                            DataPreparation.train_time_window_dimensions[3])
            job_size += second_window_size_in_minutes + progress_adjustment_gen2
            # Test iterations
        for failure_time in test_failure_times:
            window_size_in_minutes = \
                DataPreparation.calculate_time_window_delta(failure_time,
                                                            DataPreparation.test_time_window_dimensions[0],
                                                            DataPreparation.test_time_window_dimensions[1])
            job_size += window_size_in_minutes + progress_adjustment_gen2
        # Val iterations.  NOTE: Val iterations are same as test since it uses same time_window_dimension
        for failure_time in val_failure_times:
            window_size_in_minutes = \
                DataPreparation.calculate_time_window_delta(failure_time,
                                                            DataPreparation.test_time_window_dimensions[0],
                                                            DataPreparation.test_time_window_dimensions[1])
            job_size += window_size_in_minutes + progress_adjustment_gen2
        return job_size

    @staticmethod
    def calculate_time_window_delta(failure_time, start_offset, end_offset):
        windows_start = failure_time - pd.Timedelta(
            seconds=60 * start_offset)  # mins before the failure time
        windows_end = failure_time - pd.Timedelta(
            seconds=60 * end_offset)  # mins before the failure time
        time_delta = (windows_end - windows_start).total_seconds() / 60
        return time_delta
