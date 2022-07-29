from csv import DictReader
import time

from gpg import Data
from services.anomaly_data_service import AnomalyDataService
from services.synthesize_data import Data_Synthesizer

class SynthesizeDataManager:
    """Used as a  data source that periodically yields timeseries data points

    """
    @staticmethod
    def csv_line_reader(file_name, col_name):
        """Use data from a csv to periodically yield a row of data

        :param file_name: Name of csv file as source of data
        :param col_name:  Name of column to extract
        :return: none
        ..notes:: This static method has no return.  Instead, it yields a row of data that has been read from
        a data source.
        """
        with open(file_name, 'r') as read_obj:
            dict_reader = DictReader(read_obj)
            for row in dict_reader:
                # print("row in reader: {}".format(row))
                time.sleep(1 / 10)
                yield [row['timestamp'], row[col_name]]

    @staticmethod
    def load_sensor(col_name):
        query = AnomalyDataService
        df_data = query.get_all_data()
        
        df_sensor = df_data[['sensortimestamp', col_name]]
        for index in df_sensor.index:
                # print("row in reader: {}".format(row))
                row = df_sensor.loc[index,:]
                time.sleep(1 / 10)
                yield [row['sensortimestamp'], row[col_name]]

    @staticmethod
    def synthesize_data(col_name):
        generator = Data_Synthesizer
        df_data = generator.synthesize_data(col_name)
        
        df_sensor = df_data[['timestamp', col_name]]
        for index in df_sensor.index:
                # print("row in reader: {}".format(row))
                row = df_sensor.loc[index,:]
                print(row)
                time.sleep(1 / 10)
                yield [row['timestamp'], row[col_name]]

