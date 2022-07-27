import numpy as np

import pandas as pd
class SensorDataServiceCSV:
    base_path = 'static'
    csv_name = 'sensor.csv'
    X_train_name = 'X_train.npy'
    Y_train_name = 'Y_train.npy'
    X_test_name = 'X_test.npy'
    Y_test_name = 'Y_test.npy'
    X_val_name = 'X_val.npy'
    Y_val_name = 'Y_val.npy'

    df_test_name = 'DF_test.npy'
    df_val_name = 'DF_val.npy'

    DATA_TYPE_TRAIN = 1
    DATA_TYPE_TEST = 2
    DATA_TYPE_VAL = 3
    data_dict = {'TRAIN' : [X_train_name, Y_train_name], 'TEST': [X_test_name, Y_val_name], 'VAL': [X_val_name, Y_val_name]}

    @staticmethod
    def get_all_sensor_data():
        df = pd.read_csv(SensorDataServiceCSV.base_path  + '/' + SensorDataServiceCSV.csv_name, index_col='timestamp', parse_dates=True)
        return df
    @staticmethod
    def save_csv(ndarray_data, ndarray_name):
        file_path = SensorDataServiceCSV.base_path + '/' + ndarray_name
        ndarray_data.tofile(file_path, sep=',')

    @staticmethod
    def save_df(df, df_file_name):
        file_path = SensorDataServiceCSV.base_path + '/' + df_file_name
        df.to_csv(file_path)

    @staticmethod
    def get_csv_as_df(df_file_name):
        return pd.read_csv(SensorDataServiceCSV.base_path + '/' + df_file_name, index_col='timestamp', parse_dates=True)

    @staticmethod
    def get_csv_as_array(ndarray_name):
        with open(SensorDataServiceCSV.base_path + '/' + ndarray_name) as file_name:
            nparray = np.loadtxt(file_name, delimiter=',')
        return nparray

    # :param type_of_data Can be one of 'TRAIN', 'TEST' or 'VAL'
    @staticmethod
    def getXY_data(type_of_data):
        file_names_array = SensorDataServiceCSV.data_dict.get(type_of_data)
        X = SensorDataServiceCSV.get_csv_array(file_names_array[0])
        y = SensorDataServiceCSV.get_csv_array(file_names_array[1])
        return X, y

    # Save data in binary form.
    # :param data Either ndarray or dataframe
    # :param file_name One of the class variables:  X_train_name, Y_train_name, ... df_test_name
    @staticmethod
    def save_data(data, file_name):
        file_path = SensorDataServiceCSV.base_path + '/' + file_name
        np.save(file_path, data)

    # Load data from binary file
    # :param file_name One of the class variables:  X_train_name, Y_train_name, ... df_test_name
    @staticmethod
    def load_data(file_name):
        file_path = SensorDataServiceCSV.base_path + '/' + file_name
        data = np.load(file_path)
        return data


