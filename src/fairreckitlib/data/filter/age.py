"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
from .count import CountFilter
from .base import DataFilter


class AgeFilter(CountFilter):
    """Filters the dataframe on user age, if such a column exists."""

    def __init__(self, dataset: pd.DataFrame, min_val: int = 0, max_val: int = 100) -> None:
        """
        The constructor.

        Args:
            min_val: minimum count (default 0)
            max_val: maximum count (default 100)
        """

        super().__init__(dataset)
        self.min_val = min_val
        self.max_val = max_val

    def __str__(self):
        """To string

        Returns:
            name of the class
        """

        return self.__class__.__name__


def create_age_filter(data_frame: pd.DataFrame, min_val: int = 0, max_val: int = 100
                     ) -> DataFilter:
    """Create an instance of the class AgeFilter

    Args:
        data_frame: a pandas DataFrame being filtered
        min_val: minimum count (default 0)
        max_val: maximum count (default 100)

    Returns:
        an instance of the AgeFilter class
    """

    return AgeFilter(data_frame, min_val, max_val)
