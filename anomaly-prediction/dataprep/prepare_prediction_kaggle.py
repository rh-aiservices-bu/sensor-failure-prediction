"""
Slice Kaggle data found in sensor.csv into partitions that start 24 hours before each failure.  All columns are saved.
These partitions are used to simulate real time data that is generated for model prediction.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../static/sensor.csv', index_col='timestamp', parse_dates=True)
# Create new df with 4 sensors, the target,, 'machine_status' and same index


# Convert string values in target with numerics
status_values = [(df['machine_status'] == 'NORMAL'), (df['machine_status'] == 'BROKEN'),
                         (df['machine_status'] == 'RECOVERING')]
numeric_status_values = [0, 1, 0.5]
df['machine_status'] = np.select(status_values, numeric_status_values, default=0)

# Get failure times
failure_times = df[df['machine_status'] == 1].index



for i, failure_time in enumerate(failure_times):
    df.loc[(failure_time - pd.Timedelta(seconds=60*60*24)) : failure_time, :].\
        to_csv('../static/kaggle_prediction_data/prediction_slice'+str(i)+'.csv')
