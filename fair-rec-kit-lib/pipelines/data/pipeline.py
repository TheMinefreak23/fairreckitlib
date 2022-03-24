""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

'''
1. load in the dataset using .tsv files
2. aggregate the dataset (optional) (should not be fully implemented yet)
3. convert the ratings (should not be fully implemented yet)
4. split the dataset into train/test using either a set ratio, random, or timestamps
5. return the .tsv files so the model pipeline can load in the train and test .tsv files
'''




from abc import ABCMeta, abstractmethod
import time
import pandas as pd
import callback as cb

import sys

sys.path.append('..\\H_repo_lib\\dataloaders')
import dataloaders as dl


'''
This class contains all the necessary functions to run the entire data pipeline.
From loading in the required dataset(s) to aggregating them, converting the ratings, 
splitting it into train/test set, and saving these in the designated folder.
'''
class DataPipeline(metaclass=ABCMeta):

    def __init__(self):
        pass

    '''
    Runs the entire data pipeline
    '''
    def run(self, df_name, dest_folder_path, ratio, filters, callback, **args):
        callback.on_begin_pipeline()

        start = time.time()
        self.load_df(df_name, callback)
        self.aggregate(filters, callback)
        self.convert(callback)
        self.split(ratio, callback)
        self.save_sets(dest_folder_path, callback)
        end = time.time()

        callback.on_end_pipeline(end - start)

    '''
    Loads in the desired dataset using the dataloader function.
    This function returns a dictionary containing the dataframe and metadata. 
    '''
    def load_df(self, df_name, callback):
        callback.on_begin_load_df(df_name)

        start = time.time()
        dl.dataloader(df_name)
        end = time.time()

        callback.on_end_load_df(end - start)

    def aggregate(self, filters, callback):
        callback.on_begin_aggregate(filters)

        start = time.time()
        # aggregated the set using the given filters
        end = time.time()

        callback.on_end_aggregate(end - start)

    def convert(self, callback):
        callback.on_begin_convert()

        start = time.time()
        # convert the ratings of the dataset
        end = time.time()

        callback.on_end_convert(end - start)

    def split(self, ratio, callback):
        callback.on_begin_split(ratio)

        start = time.time()
        # split the dataset into train&test using the given ratio
        end = time.time()

        callback.on_end_split(end - start)

    def save_sets(self, dest_folder_path, callback):
        callback.on_saving_sets(dest_folder_path)

        start = time.time()
        # save the train and test sets to the given destination
        end = time.time()

        callback.on_saved_sets(dest_folder_path, end - start)


# LINES BELOW ONLY FOR TESTING
# DELETE LATER

callback = cb.DataPipelineConsole()
dp = DataPipeline()
dp.run('ml_100k_u', '..\\Datasets\\', (80, 20), ['gender', 'age'], callback)
