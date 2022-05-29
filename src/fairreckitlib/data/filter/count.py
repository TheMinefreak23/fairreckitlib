"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
import pandas as pd
from .base import DataFilter


class CountFilter(DataFilter):
    """Filters the dataframe on user play count, if such a column exists."""

    def run(self, col_name: str, min_val: int = 0, max_val: int = math.inf) -> pd.DataFrame:
        """
        Filter the dataframe based on (play_)count or rating column in the range of
        min_val and max_val values.

        Args:
            col_name: the name of the column to be filtered
            min_val: minimum count (default 0)
            max_val: maximum count (default 0)

        Returns:
            a filtered dataframe from the given dataframe
        """
        if col_name in self.dataset.columns:
            df_filter = self.dataset[col_name].between(min_val, max_val, inclusive="both")
            return self.dataset[df_filter].reset_index(drop=True)
        return self.dataset

    def __str__(self):
        """To string

        Returns:
            name of the class
        """
        return self.__class__.__name__


def create_count_filter(data_frame: pd.DataFrame) -> DataFilter:
    """Create an instance of the class CountFilter

    Args:
        data_frame: a pandas DataFrame being filtered

    Returns:
        an instance of the CountFilter class
    """
    return CountFilter(data_frame)
