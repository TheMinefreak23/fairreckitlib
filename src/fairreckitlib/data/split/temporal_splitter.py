"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import lenskit.crossfold as xf

from .base_splitter import DataSplitter


class TemporalSplitter(DataSplitter):
    """Temporal Splitter.

    Splits the dataframe into a train and test set based on time.
    """

    def run(self, dataframe, test_ratio):
        # Note: for this function to work, it needs a 'user' and 'timestamp' column.
        frac = xf.LastFrac(test_ratio, col='timestamp')
        for train_set, test_set in xf.partition_users(dataframe, 1, frac):
            return train_set, test_set
