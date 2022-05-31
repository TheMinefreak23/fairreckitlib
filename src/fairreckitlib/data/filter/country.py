"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import List
import pandas as pd
from .base import DataFilter
from .gender import GenderFilter

class CountryFilter(GenderFilter):
    """Filters the dataframe on country, if such column exists."""

    def run(self, col_name: str) -> pd.DataFrame:
        """Filter specific country of the dataframe.

        Args:
            col_name: the name of the column to be filtered

        Returns:
            a filtered dataframe from the given dataframe
        """
        if col_name in self.dataset.columns and self.filters:
            df_filter = []
            for country in self.filters:
                df_filter.append(self.dataset[col_name]
                                 .map(lambda x: x.lower() if x else x
                                     ).eq(country.lower() if country else country))
            return self.do_filtering(df_filter)
        return self.dataset

    def __str__(self):
        """To string

        Returns:
            name of the class
        """
        return self.__class__.__name__


def create_country_filter(data_frame: pd.DataFrame, filters: List[str] = None) -> DataFilter:
    """Create an instance of the class CountryFilter

    Args:
        data_frame: a pandas DataFrame being filtered
        filters: the list of filters used in filtering

    Returns:
        an instance of the CountryFilter class
    """
    return CountryFilter(data_frame, filters)
