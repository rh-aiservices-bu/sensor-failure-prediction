import pandas as pd
import numpy as np
import random
from pickle import dump, load
import datetime
import time

import torch

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

class Data_Synthesizer:

    @staticmethod 
    def synthesize_data(sensor_name=''):
        config = DGANConfig(
            max_sequence_len=100,
            sample_len=20, # trying a larger sample_len
            batch_size=min(1000, 51),
            apply_feature_scaling=True, 
            apply_example_scaling=False,
            use_attribute_discriminator=False,
            generator_learning_rate=1e-4,
            discriminator_learning_rate=1e-4,
            epochs=10000)

        model = DGAN(config)

        # loading model for future use 
        model = model.load("static/dgan_casing.pt")
        # Generate synthetic data - this ran near instantly
        _, synthetic_features = model.generate_numpy(1000)

        reshaped_data = np.empty((100000,2))
        init = 0 

        # outer loop around samples
        for x in range(1000):
            # loop of each sample
            for y in range(100):
                reshaped_data[init][0] = 1
                reshaped_data[init][1] = synthetic_features[x,y,0]
                init+=1
        if sensor_name == '':
            sensor_name = 'sensor_' + str(random.randint(111,999))
        
        reshaped_df = pd.DataFrame(reshaped_data, columns=["timestamp", sensor_name])

        date_from = datetime.datetime.strptime('2017-04-08', '%Y-%m-%d')
        dates = []
        for row in range(100000):
            dates.append(str(date_from + datetime.timedelta(hours=(3*row))))

        reshaped_df['timestamp'] = dates
        return reshaped_df

    
