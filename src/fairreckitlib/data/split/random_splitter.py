"""This module contains random splitting functionality.

Classes:

    RandomSplitter: can split randomly.

Functions:

    create_random_splitter: create an instance of the class.


This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Dict, Tuple

import lenskit.crossfold as xf
import pandas as pd
from seedbank import numpy_rng

from .base_splitter import DataSplitter


class RandomSplitter(DataSplitter):
    """Random Splitter.

    Splits the dataframe into a train and test set randomly.
    """

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into a train and test set.

        Args:
            dataframe: with at least the 'user' column.

        Returns:
            the train and test set dataframes of the split.
        """
        rng_spec = numpy_rng(spec=self.params['seed'])
        frac = xf.SampleFrac(self.test_ratio)
        for train_set, test_set in xf.partition_users(dataframe, 1, frac, rng_spec=rng_spec):
            return train_set, test_set


def create_random_splitter(name: str, params: Dict[str, Any], **kwargs) -> RandomSplitter:
    """Create the Random Splitter.

    Returns:
        the random data splitter.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return RandomSplitter(name, params, kwargs['test_ratio'])
