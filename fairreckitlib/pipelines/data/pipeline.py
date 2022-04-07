""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)


1. load in the dataset using .tsv files
2. aggregate the dataset (optional) (should not be fully implemented yet)
3. convert the ratings (should not be fully implemented yet)
4. split the dataset into train/test using either a set ratio, random, or timestamps
5. return the .tsv files so the model pipeline can load in the train and test .tsv files
"""

import os
import sys
import time

from abc import ABCMeta, abstractmethod
from fairreckitlib.experiment.common import EXP_KEY_DATASET_SPLIT_PARAMS
from fairreckitlib.experiment.common import EXP_KEY_DATASET_SPLIT_TEST_RATIO
from fairreckitlib.experiment.common import EXP_KEY_DATASET_SPLIT_TYPE


class DataPipeline(metaclass=ABCMeta):
    """This class contains all the necessary functions to run the entire data pipeline.

    From loading in the required dataset(s) to aggregating them, converting the ratings,
    splitting it into train/test set, and saving these in the designated folder.
    """

    def __init__(self, split_factory):
        self.split_datasets = dict()
        self.split_factory = split_factory

    def run(self, dst_dir, dataset, prefilters, rating_modifier, split, callback, **kwargs):
        """Runs the entire data pipeline by calling all functions of the class in order."""
        callback.on_begin_pipeline()

        start = time.time()
        data_dir = self.create_data_dir(dst_dir, dataset.name)
        df = self.load_df(dataset, callback)
        df = self.aggregate(df, prefilters, callback)
        df = self.convert(df, rating_modifier, callback)
        rating_scale = (df['rating'].min(), df['rating'].max())
        rating_type = dataset.ratings
        train_set, test_set = self.split(df, split, callback)
        train_path, test_path = self.save_sets(
            train_set[['user', 'item', 'rating']],
            test_set[['user', 'item', 'rating']],
            data_dir,
            callback
        )
        end = time.time()

        callback.on_end_pipeline(end - start)

        return data_dir, train_path, test_path, rating_scale, rating_type

    def create_data_dir(self, dst_dir, dataset_name):
        if not self.split_datasets.get(dataset_name):
            self.split_datasets[dataset_name] = 0

        index = self.split_datasets[dataset_name]
        data_dir = os.path.join(dst_dir, dataset_name + '_' + str(index))
        os.mkdir(data_dir)
        self.split_datasets[dataset_name] += 1

        return data_dir

    def load_df(self, dataset, callback):
        """Loads in the desired dataset using the dataloader function.

        Returns a dictionary containing the pandas dataframe(s) belonging to the given dataset.
        """
        callback.on_begin_load_df(dataset.name)

        start = time.time()
        df = dataset.load_matrix_df()
        end = time.time()

        callback.on_end_load_df(end - start)

        return df

    def aggregate(self, df, prefilters, callback):
        """Aggregates the dataframe using the given filters."""
        if len(prefilters) == 0:
            return df

        callback.on_begin_aggregate(prefilters)

        start = time.time()
        # TODO aggregated the set using the given filters
        for filter in prefilters:
            continue
        end = time.time()

        callback.on_end_aggregate(end - start)

        return df

    def convert(self, df, rating_modifier, callback):
        """Converts the ratings in the dataframe to be X"""
        if rating_modifier is None:
            return df

        callback.on_begin_convert()

        start = time.time()
        # TODO convert the ratings of the dataset
        end = time.time()

        callback.on_end_convert(end - start)

        return df

    def split(self, df, split_config, callback):
        """Splits the dataframe into a train and test set.

        This will be split 80/20 (or a similar ratio), and be done either random, or timestamp-wise.
        """
        test_ratio = split_config[EXP_KEY_DATASET_SPLIT_TEST_RATIO]
        callback.on_begin_split(test_ratio)

        start = time.time()
        splitter = self.split_factory[split_config[EXP_KEY_DATASET_SPLIT_TYPE]]()
        params = split_config[EXP_KEY_DATASET_SPLIT_PARAMS]
        train_set, test_set = splitter.run(df, test_ratio, params)
        end = time.time()

        callback.on_end_split(end - start)

        return train_set, test_set

    def save_sets(self, train_set, test_set, data_dir, callback):
        """Saves the train and test sets to the desired folder."""
        callback.on_saving_sets(data_dir)

        train_path = os.path.join(data_dir, 'train_set.tsv')
        test_path = os.path.join(data_dir, 'test_set.tsv')

        start = time.time()
        train_set.to_csv(train_path, sep='\t', header=False, index=False)
        test_set.to_csv(test_path, sep='\t', header=False, index=False)
        end = time.time()

        callback.on_saved_sets(data_dir, end - start)

        return train_path, test_path
