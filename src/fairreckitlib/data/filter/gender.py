
"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from functools import reduce
from typing import List
import pandas as pd
from .base import DataFilter


class GenderFilter(DataFilter):
    """Filters the dataframe on gender column f/m, if such a column exists."""

    def __init__(self, dataset: pd.DataFrame, filters: List[str] = None) -> None:
        """
        The constructor.

        Args:
            filters: the list of filters used in filtering
        """
        super().__init__(dataset)
        self.filters = filters

    def run(self, col_name: str) -> pd.DataFrame:
        """Filter specific gender of the dataframe.

        Args:
            col_name: the name of the column to be filtered

        Returns:
            a filtered dataframe from the given dataframe
        """

        if col_name in self.dataset.columns and self.filters:
            df_filter = []
            for gender in self.filters:
                gender = gender.lower()
                if gender in ['female', 'f']:
                    df_filter.append((self.dataset[col_name].map(lambda x: x.lower() if x else x
                                                                ).eq('f') |
                                    self.dataset[col_name].map(lambda x: x.lower() if x else x
                                                              ).eq('female')))
                elif gender in ['male', 'm']:
                    df_filter.append((self.dataset[col_name].map(lambda x: x.lower() if x else x
                                                                ).eq('male') |
                                      self.dataset[col_name].map(lambda x: x.lower() if x else x
                                                                ).eq('m')))
                else: df_filter.append((self.dataset[col_name].map(lambda x: x.lower() if x else x
                                                                  ).eq(gender)))
            return self.do_filtering(df_filter)
        return self.dataset

    def do_filtering(self, df_filter: List[List[bool]]) -> pd.DataFrame:
        """
        Combining all the filtering conditions and filter the dataframe

        Args:
            df_filter: the list of all filtering conditions

        Returns:
            a filtered dataframe from the given dataframe
        """
        if df_filter:
            df_filter = reduce(lambda x, y: (x) | (y), df_filter)
            return self.dataset[df_filter].reset_index(drop=True)
        return self.dataset

    def __str__(self):
        """To string

        Returns:
            name of the class
        """
        return self.__class__.__name__

def create_gender_filter(data_frame: pd.DataFrame, filters: List[str] = None) -> DataFilter:
    """Create an instance of the class GenderFilter

    Args:
        data_frame: a pandas DataFrame being filtered
        filters: the list of filters used in filtering

    Returns:
        an instance of the GenderFilter class
    """
    return GenderFilter(data_frame, filters)
