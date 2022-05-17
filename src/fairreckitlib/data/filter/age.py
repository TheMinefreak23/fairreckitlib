"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
from .base import DataFilter


class AgeFilter(DataFilter):
    """
    Filters the dataframe on user age, if such a column exists.
    """
    def run(self, min_val: int = 0, max_val: int = 100) -> pd.DataFrame:
        """
        Filters the dataframe based on age column in the range of min_val and max_val values.

        Args:
            min_val: minimum age (default 0)
            max_val: maximum age (default 0)

        Returns:
            a filtered dataframe from the given dataframe
        """
        if 'age' in self.dataset.columns:
            df_filter = self.dataset.age.between(min_val, max_val, inclusive="both")
            return self.dataset[df_filter]
        return self.dataset

    def __str__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__class__.__name__


def create_age_filter(data_frame: pd.DataFrame) -> DataFilter:
    """
    Creates an instance of the class AgeFilter

    Args:
        data_frame: a pandas DataFrame being filtered

    Returns:
        an instance of the AgeFilter class
    """
    return AgeFilter(data_frame)
