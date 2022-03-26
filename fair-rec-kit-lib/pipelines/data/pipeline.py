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
import lenskit.crossfold as xf


import callback as cb

import sys

sys.path.append('..\\H_repo_lib\\dataloaders')
import dataloaders as dl



class DataPipeline(metaclass=ABCMeta):
    '''This class contains all the necessary functions to run the entire data pipeline.

    From loading in the required dataset(s) to aggregating them, converting the ratings, 
    splitting it into train/test set, and saving these in the designated folder.
    '''
    def __init__(self):
        pass

    
    def run(self, df_name, dest_folder_path, ratio, time_split, filters, callback, **args):
        '''Runs the entire data pipeline by calling all functions of the class in order.'''
        callback.on_begin_pipeline()

        start = time.time()
        df = self.load_df(df_name, callback)
        self.aggregate(df, filters, callback)
        self.convert(df, callback)
        tt_pairs = self.split(df, ratio, time_split, callback)
        self.save_sets(tt_pairs, dest_folder_path, callback)
        end = time.time()

        callback.on_end_pipeline(end - start)

    
    def load_df(self, df_name, callback):
        '''Loads in the desired dataset using the dataloader function.

        This function returns a dictionary containing the pandas dataframe(s) belonging to the given dataset. 
        '''
        callback.on_begin_load_df(df_name)

        start = time.time()
        # There is a bug that needs to be solved before the dataloader function can be used here
        #df_dict = dl.dataloader(df_name)
        #df = df_dict['sub_dataset']
        df = pd.read_csv('..\\Datasets\\ml-100k\\u.data', delimiter='\t', engine='python')
        end = time.time()

        callback.on_end_load_df(end - start)

        return df


    def aggregate(self, df, filters, callback):
        '''Aggregates the dataframe using the given filters.'''
        callback.on_begin_aggregate(filters)

        start = time.time()
        # TODO aggregated the set using the given filters
        end = time.time()

        callback.on_end_aggregate(end - start)

        return df

    
    def convert(self, df, callback):
        '''Converts the ratings in the dataframe to be X'''
        callback.on_begin_convert()

        start = time.time()
        # TODO convert the ratings of the dataset
        end = time.time()

        callback.on_end_convert(end - start)

        return df

    
    def split(self, df, ratio, time_split, callback):
        '''Splits the dataframe into a train and test set.
        
        This will be split 80/20 (or a similar ratio), and be done either random, or timestamp-wise.
        '''
        callback.on_begin_split(ratio)

        start = time.time()
        # TODO split the dataset into train&test using the given ratio and do it random or time-wise
        tt_pairs = xf.partition_rows(df, 2, rng_spec=None)
        end = time.time()

        callback.on_end_split(end - start)

        return tt_pairs

    
    def save_sets(self, tt_pairs, dest_folder_path, callback):
        '''Saves the train and test set to the desired folder.'''
        callback.on_saving_sets(dest_folder_path)

        start = time.time()
        i = 0
        for (train, test) in tt_pairs:
          train.to_csv(dest_folder_path + 'train' + str(i) + '.tsv', sep='\t')
          test.to_csv(dest_folder_path + 'test' + str(i) + '.tsv', sep='\t')
          i += 1
        end = time.time()

        callback.on_saved_sets(dest_folder_path, end - start)


# LINES BELOW ONLY FOR TESTING
# DELETE LATER

callback = cb.DataPipelineConsole()
dp = DataPipeline()
dp.run('ml_100k_u', '..\\Datasets\\', (80, 20), False, ['gender', 'age'], callback)