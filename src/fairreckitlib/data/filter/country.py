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
            df_filter = self.dataset.country.map(lambda x: x.lower() if x else x
                                                ).eq(country.lower() if country else country)
            return self.dataset[df_filter].reset_index(drop=True)
        return self.dataset

    def __str__(self):
        """
        to string

        Returns:
            name of the class
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
