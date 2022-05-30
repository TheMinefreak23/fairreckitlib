"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict
import pandas as pd
from .base import DataFilter


class NumericalFilter(DataFilter):
    """Filters the dataframe on numerical data, such as age or rating.
    
    Public method:
        filter
    """

    def filter(self, dataframe: pd.DataFrame, column_name: str, min, max) -> pd.DataFrame:
        """Filters the dataframe on values in the range of min and max.

        Args:
            dataframe: Dataframe to be filtered on.
            column_name: Name of the column where the conditions need to be met.
            min: Minimal number.
            max: Maximum number.

        Returns:
            A filtered dataframe.
        """
        df_filter = dataframe[column_name].between(min, max, inclusive="both")
        return dataframe[df_filter].reset_index(drop=True)

    def __filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Private filter used in run(). Requires configuration file."""
        return self.filter(dataframe, self.params['name'], self.params["min"], self.params["max"])


def create_numerical_filter(name: str, 
                      params: Dict[str, Any], 
                      **kwargs) -> DataFilter:
    """Create an instance of the class NumericalFilter.

    Args:
        name: Name of the filter.
        params: Configuration file.
        **kwargs: Contains dataset and matrix_name.

    Returns:
        An instance of the NumericalFilter class.
    """
    return NumericalFilter(name, params, kwargs)
