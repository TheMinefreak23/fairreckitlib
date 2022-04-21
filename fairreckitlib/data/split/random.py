"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import lenskit.crossfold as xf

from .splitter import DataSplitter


class RandomSplitter(DataSplitter):
    """Random Splitter.

    Splits the dataframe into a train and test set randomly.
    """

    def run(self, dataframe, test_ratio):
        frac = xf.SampleFrac(test_ratio)
        for train_set, test_set in xf.partition_users(dataframe, 1, frac):
            return train_set, test_set
