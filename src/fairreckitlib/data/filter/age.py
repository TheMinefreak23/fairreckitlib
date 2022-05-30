"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict
import pandas as pd
from .base import DataFilter


class AgeFilter(DataFilter):
    """Filters the dataframe on user age, if such a column exists."""

    def __filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataframe based on age column in the range of min_val and max_val values.

        Args:
            min_val: minimum age (default 0)
            max_val: maximum age (default 0)

        Returns:
            a filtered dataframe from the given dataframe
        """
        if 'age' in dataframe.columns:
            df_filter = dataframe.age.between(self.params["min"], self.params["max"], inclusive="both")
            return dataframe[df_filter].reset_index(drop=True)
        return dataframe


def create_age_filter(name: str, 
                      params: Dict[str, Any], 
                      **kwargs) -> DataFilter:
    """Create an instance of the class AgeFilter

    Args:
        name: UserAge
        params: Dictionary with 'min' and 'max'.
        **kwargs (Optional): Not used.

    Returns:
        an instance of the AgeFilter class
    """
    # If params == None, thus no param in factory.create() has been given.
    if not params:  # Default value
        params = {"min": 0, "max": 100}
    return AgeFilter(name, params, kwargs)
