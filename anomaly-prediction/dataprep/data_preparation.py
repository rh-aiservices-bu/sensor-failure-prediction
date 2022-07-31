from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from utils.data_type_enum import DataTypeEnum
import joblib


from services.sensor_data_service import SensorDataServiceCSV


class DataPreparation:
    # Class variables for easy retrieval of data
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

    def __init__(self):
        DataPreparation.do_automated_data_prep()


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
        """Drop unusable columns

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
        Replace NaN columnwise with mean of each column
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
            df.loc[start_predic_time:stop_predic_time, 'alarm'] = 2  # can not use 1, because 1 indicates the machine failure time

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
    def get_scaler(training_data, sensor_names):
        """Get a scaler for training data
        Apply MinManScaler to the training data using the range (0, 1)
        :param training_data:
        :type: Pandas DataFrame
        :param sensor_names: List of sensor names(feature names)
        :type: list
        :return: scaler for the training data
        :type: MinMaxScaler
        """
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = min_max_scaler.fit(training_data[sensor_names])

        return scaler


    @staticmethod
    def scale_dataframe(scaler, data_df, sensor_names):
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
        num_components = fit_pca.components_.shape[0]
        most_important_indexes = [np.abs(fit_pca.components_[i]).argmax() for i in range(num_components)]
        most_important_names = [feature_names[most_important_indexes[i]] for i in range(num_components)]
        # bundle ranked important feature names with variance ratio
        name_dict = {most_important_names[i]: fit_pca.explained_variance_ratio_[i] for i in range(num_components)}

        df_ranked_features = pd.DataFrame(name_dict.items())
        df_ranked_features.columns = ['Ranked Features', 'Variance Ratio']
        return df_ranked_features
    @staticmethod
    def transform_df_by_pca(pca, df_data, scaled_data, num_features_to_include, sensor_names):
        """Reduce the number of features in the training data by the parameter num_features_to_include.

        :param pca: The PCA for the training data
        :type: PCA
        :param df_data: Training data dataframe
        :type: Pandas DataFrame
        :param scaled_data: Array of scaled data
        :type: ndarray
        :param num_features_to_include: Number of features to include in training
        :type: int
        :param sensor_names: List of sensor names that will be included
        :type: List of string
        :return: Training data that has been scaled and whose dimensions have been reduced
        :type: Pandas DataFrame
        """

        data_transformed = pca.transform(scaled_data)  # ndarray
        df_transformed = pd.DataFrame(data_transformed)
        # number of components
        n_features = pca.components_.shape[0]

        # get the index of the most important feature on EACH component i.e. largest absolute value
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_features)]

        # get the names
        most_important_names = [sensor_names[most_important[i]] for i in range(n_features)]
        no_duplications = list(OrderedDict.fromkeys(most_important_names))
        no_duplications = no_duplications[0:(num_features_to_include)]

        # Create a map between pca indexes (0,1,2,.....n_features) and most_important_names
        dict1 = dict(zip(range(n_features), most_important_names))

        df_transformed['machine_status'] = df_data['machine_status'].values
        df_transformed['alarm'] = df_data['alarm'].values
        df_transformed.index = df_data.index
        df_transformed.rename(columns=dict1, inplace=True)
        df_smaller = df_transformed[no_duplications]
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
    def timeseries_before_failure( df, failure_times, feature_names,
                                  timewindow_dimensions, window_len, stride, data_type):

        # [96*60, 12*60, 12*60, 5]
        '''
        Generate data samples using the time windows ahead of each machine failure time;
        window_len: how many data points from each feature will be used to make one sample for the model.
        stride: sliding window size
        Transform original df from a 2D array [samples, features] to a 3D array [samples, timesteps, features*window_len]
        '''
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
            time_delt_min = windows_end - windows_start
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
        id_keep = [i for i, x in enumerate(Y) if (x == 1) or (x == 0)]
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
    def make_predict_data( df, failure_times, feature_names,
                                  timewindow_dimensions, window_len, stride, data_type):

        # [96*60, 12*60, 12*60, 5]
        '''
        Generate data samples using the time windows ahead of each machine failure time;
        window_len: how many data points from each feature will be used to make one sample for the model.
        stride: sliding window size
        Transform original df from a 2D array [samples, features] to a 3D array [samples, timesteps, features*window_len]
        '''

        stride = 1
        X = np.empty((1, 1, window_len * len(feature_names)),
                     float)  # [samples, timesteps, features].  Samples will be appended below
        Y = np.empty((1), float)

        # For each failure_time, generate two arrays of data. First array starts well (timewindow_for_use[0]) before failure time
        # and ends short(timewindow_for_use[1]) before the failure.
        # Second time array starts where first array ends(timewindow_for_use[2] and ends close to the failure (timewindow_for_use[3])
        for i, failure_time in enumerate(failure_times):
            windows_start = failure_time - pd.Timedelta(
                seconds=60 * timewindow_dimensions[0])  # mins before the failure time
            windows_end = failure_time - pd.Timedelta(
                seconds=60 * timewindow_dimensions[1])  # mins before the failure time
            time_delt_min = windows_end - windows_start
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

        # remove samples where y is neither 0 nor 1
        id_keep = [i for i, x in enumerate(Y) if (x == 1) or (x == 0)]
        y_data = Y[id_keep]
        X_data = X[id_keep][:, :]





    # Prepare all data to be used for train and testing.  Store the results in Class Variables for easy retrieval
    @staticmethod
    def do_automated_data_prep():
        df = DataPreparation.get_df()
        # Get a series that shows how many nulls in each column.
        count_nulls_per_column = DataPreparation.get_null_list(df)
        # print(count_nulls_per_column)
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

        sensor_names = df_train.columns.tolist()[:-2]  # Get sensor names, omitting last two columns ('machine_status', 'Operation')
        #  Get scaler that has been fit to training data
        min_max_scaler = DataPreparation.get_scaler(df_train, sensor_names)

        # Save scaler for use in prediction
        joblib.dump(min_max_scaler, 'static/training_scaler.gz')

        # Scale (transform) using the previously formed min_max_scaler.  Results are ndarray
        DataPreparation.scaled_train = DataPreparation.scale_dataframe(min_max_scaler, df_train, sensor_names)
        DataPreparation.scaled_test = DataPreparation.scale_dataframe(min_max_scaler, df_test, sensor_names)
        DataPreparation.scaled_val = DataPreparation.scale_dataframe(min_max_scaler, df_val, sensor_names)

        num_top_components = 8  # only give top 8 features
        # Get PCA df to determine which top features to include
        DataPreparation.pca = DataPreparation.get_PCA(DataPreparation.scaled_train, num_top_components)
        # Get Dataframe of ranked features determined by PCA
        DataPreparation.ranked_features = DataPreparation.get_ranked_features(DataPreparation.pca, df_train.columns)

        num_features_to_include = 4
        DataPreparation.num_features_to_include = num_features_to_include
        # transform_df_by_pca(pca, df_data, scaled_data, num_features_to_include, sensor_names):
        DataPreparation.df_train_pca = DataPreparation.transform_df_by_pca(
            DataPreparation.pca, df_train, DataPreparation.scaled_train,
            num_features_to_include, sensor_names)
        DataPreparation.df_test_pca = DataPreparation.transform_df_by_pca(
            DataPreparation.pca, df_test, DataPreparation.scaled_test,
            num_features_to_include,sensor_names)
        DataPreparation.df_val_pca = DataPreparation.transform_df_by_pca(
            DataPreparation.pca, df_val, DataPreparation.scaled_val,
            num_features_to_include, sensor_names)

        ########## Stop here.  The above code can be done, but the code that follows must be done asynchronously since
        # it is time consuming


     # Finish data prep that wasn't done in do_automated_data_prep().
    # This method should be done as an asychronous process so that the page does not get blocked.
    @staticmethod
    def finish_data_prep():



        feature_names = DataPreparation.df_train_pca.columns.tolist()[:-2]

        train_failure_times = DataPreparation.get_failure_times(DataPreparation.df_train_pca)
        test_failure_times = DataPreparation.get_failure_times(DataPreparation.df_test_pca)
        val_failure_times = DataPreparation.get_failure_times(DataPreparation.df_val_pca)

        DataPreparation.job_size = DataPreparation.calculate_job_size(train_failure_times, test_failure_times, val_failure_times)

        print("Calculated job size:  {}".format(DataPreparation.job_size))

        DataPreparation.progress_counter = 0
        yield "event: initialize\ndata: " + str(DataPreparation.job_size) + "\n\n"



        # Train windows use DataPreparation.stride for first windows, then stride=1 for second group.

        yield from DataPreparation.timeseries_before_failure(DataPreparation.df_train_pca, train_failure_times, feature_names,
                                                     DataPreparation.train_time_window_dimensions,
                                                     DataPreparation.window_size, DataPreparation.stride,
                                                     data_type = DataTypeEnum.TRAIN)

        # Test and Val use stride = 1 by default, no matter what gets passed in the call.

        yield from DataPreparation.timeseries_before_failure(DataPreparation.df_test_pca, test_failure_times, feature_names,
                                                   DataPreparation.test_time_window_dimensions,
                                                   DataPreparation.window_size, DataPreparation.stride,
                                                   data_type = DataTypeEnum.TEST)

        # Use same window dimensions for test and val

        yield from DataPreparation.timeseries_before_failure(DataPreparation.df_val_pca, val_failure_times, feature_names,
                                                     DataPreparation.test_time_window_dimensions,
                                                     DataPreparation.window_size, DataPreparation.stride,
                                                     data_type = DataTypeEnum.VAL)

        yield "event: jobfinished\ndata: " + "\n\n"



        """
                num_time_steps_for_input = 1
                num_time_steps_for_output = 1
                df_train_for_supervised = DataPreparation.series_to_supervised(df_train_pca, num_time_steps_for_input, num_time_steps_for_output)
                df_test_for_supervised =  DataPreparation.series_to_supervised(df_test_pca, num_time_steps_for_input, num_time_steps_for_output)
                df_val_for_supervised =   DataPreparation.series_to_supervised(df_val_pca, num_time_steps_for_input, num_time_steps_for_output)

                df_reshaped_train = DataPreparation.reshape_data(df_train_for_supervised, num_time_steps_for_input)
                df_reshaped_test = DataPreparation.reshape_data(df_test_for_supervised, num_time_steps_for_input)
                df_reshaped_val = DataPreparation.reshape_data(df_val_for_supervised, num_time_steps_for_input)
                """

    @staticmethod
    def calculate_job_size(train_failure_times, test_failure_times, val_failure_times):
        progress_adjustment_gen1 = -3
        progress_adjustment_gen2 = -19
        job_size = 0
        # Training iterations
        for failure_time in train_failure_times:
            first_window_size_in_minutes = \
                DataPreparation.calculate_time_window_delta(failure_time, DataPreparation.train_time_window_dimensions[0],
                                                            DataPreparation.train_time_window_dimensions[1]) / 5
            job_size += first_window_size_in_minutes + progress_adjustment_gen1
            second_window_size_in_minutes = \
                DataPreparation.calculate_time_window_delta(failure_time, DataPreparation.train_time_window_dimensions[2],
                                                            DataPreparation.train_time_window_dimensions[3])
            job_size += second_window_size_in_minutes + progress_adjustment_gen2
            # Test iterations
        for failure_time in test_failure_times:
            window_size_in_minutes =\
                DataPreparation.calculate_time_window_delta(failure_time, DataPreparation.test_time_window_dimensions[0],
                                                            DataPreparation.test_time_window_dimensions[1])
            job_size += window_size_in_minutes + progress_adjustment_gen2
        # Val iterations.  NOTE: Val iterations are same as test since it uses same time_window_dimension
        for failure_time in val_failure_times:
            window_size_in_minutes =\
                DataPreparation.calculate_time_window_delta(failure_time, DataPreparation.test_time_window_dimensions[0],
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
