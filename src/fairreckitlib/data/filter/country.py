"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict
import pandas as pd
from .base import DataFilter

class CountryFilter(DataFilter):
    """Filters the dataframe on country, if such column exists."""

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Filter specific country of the dataframe.

        Args:
            country: the name of the country used in filtering

        Returns:
            a filtered dataframe from the given dataframe
        """
        if 'country' in dataframe.columns:
            country = self.params['country']
            df_filter = dataframe.country.map(lambda x: x.lower() if x else x
                                                ).eq(country.lower() if country else country)
            return dataframe[df_filter].reset_index(drop=True)
        return dataframe

    def __str__(self):
        """To string

        Returns:
            name of the class
        """
        return self.__class__.__name__

def create_country_filter(name: str, 
                          params: Dict[str, Any], 
                          **kwargs) -> DataFilter:
    """Create an instance of the class CountryFilter

    Args:
        name: UserCountry
        params: Dictionary with only one key: 'country'.
        **kwargs (Optional): Not used.

    Returns:
        an instance of the CountryFilter class
    """
    return CountryFilter(name, params)
