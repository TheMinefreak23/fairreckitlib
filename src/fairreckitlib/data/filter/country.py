"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
from .base import DataFilter

class CountryFilter(DataFilter):
    """
    Filters the dataframe on country, if such column exists.
    """
    def run(self, country: str = None) -> pd.DataFrame:
        """
        Filters specific country of the dataframe.

        Args:
            country: the name of the country used in filtering

        Returns:
            a filtered dataframe from the given dataframe
        """
        if 'country' in self.dataset.columns:
            df_filter = self.dataset.country.lower().eq(country.lower())
            return self.dataset[df_filter]
        return self.dataset

    def __str__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__class__.__name__

def create_country_filter(data_frame: pd.DataFrame) -> DataFilter:
    """
    Creates an instance of the class CountryFilter

    Args:
        data_frame: a pandas DataFrame being filtered

    Returns:
        an instance of the CountryFilter class
    """
    return CountryFilter(data_frame)
