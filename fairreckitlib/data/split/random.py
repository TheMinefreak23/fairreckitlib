"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import lenskit.crossfold as xf

from .splitter import DataSplitter


class RandomSplitter(DataSplitter):

    def run(self, df, test_ratio, params):
        for train_set, test_set in xf.partition_users(df, 1, xf.SampleFrac(test_ratio)):
            return train_set, test_set


def create_random_splitter():
    return RandomSplitter()
