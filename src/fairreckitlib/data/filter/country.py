"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from functools import reduce
from typing import List
import pandas as pd
from .base import DataFilter

class CountryFilter(DataFilter):
    """Filters the dataframe on country, if such column exists."""

    def run(self, col_name: str, countries: List[str] = None) -> pd.DataFrame:
        """Filter specific country of the dataframe.

        Args:
            col_name: the name of the column to be filtered
            countries: the list of country names used in filtering

        Returns:
            a filtered dataframe from the given dataframe
        """

        if col_name in self.dataset.columns and countries:
            df_filter = []
            for country in countries:
                df_filter.append(self.dataset[col_name]
                                 .map(lambda x: x.lower() if x else x
                                     ).eq(country.lower() if country else country))
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


def create_country_filter(data_frame: pd.DataFrame) -> DataFilter:
    """Create an instance of the class CountryFilter

    Args:
        data_frame: a pandas DataFrame being filtered

    Returns:
        an instance of the CountryFilter class
    """
    return CountryFilter(data_frame)
