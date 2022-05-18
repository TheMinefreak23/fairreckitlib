"""This module contains time splitting functionality.

Classes:

    TemporalSplitter: can split on timestamp.

Functions:

    create_temporal_splitter: create an instance of the class (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import lenskit.crossfold as xf
import pandas as pd

from .base_splitter import DataSplitter


class TemporalSplitter(DataSplitter):
    """Temporal Splitter.

    Splits the dataframe into a train and test set based on time user-by-user.
    """

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into a train and test set.

        For this function to work, it needs a 'user' and 'timestamp' column.

        Args:
            dataframe: with at least the 'user' column.

        Returns:
            the train and test set dataframes of the split.
        """
        frac = xf.LastFrac(self.test_ratio, col='timestamp')
        for train_set, test_set in xf.partition_users(dataframe, 1, frac):
            return train_set, test_set


def create_temporal_splitter(name: str, params: Dict[str, Any], **kwargs) -> TemporalSplitter:
    """Create the Temporal Splitter.

    Returns:
        the temporal data splitter.
    """
    return TemporalSplitter(name, params, kwargs['test_ratio'])
