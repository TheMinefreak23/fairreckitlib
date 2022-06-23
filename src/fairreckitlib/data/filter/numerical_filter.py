"""Module to filter on numerical data, like age or rating.

Classes:

    NumericalFilter: Filter the dataframe on numerical data, such as age or rating.

Functions:

    create_numerical_filter: Create an instance of NumericalFilter.


This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
from typing import Any, Dict
import pandas as pd
from .base_filter import DataFilter
from .filter_constants import FILTER_NUMERICAL

class NumericalFilter(DataFilter):
    """Filters the dataframe on numerical data, such as age or rating.

    Public method:
        filter
    """

    def get_type(self) -> str:
        """Get the type of the filter.

        Returns:
            The type name of the filter.
        """
        return FILTER_NUMERICAL

    def filter(self, dataframe: pd.DataFrame, column_name='',
               min_val=0, max_val=math.inf) -> pd.DataFrame:
        """Filter the dataframe on values in the range of min_val and max_val.

        Args:
            dataframe: Dataframe to be filtered on.
            column_name (str): Name of the column where the conditions need to be met.
            min_val (int | float): Minimal number (default 0).
            max_val (int | float): Maximum number (default infinite).

        Returns:
            A filtered dataframe.
        """
        if column_name not in dataframe.columns:
            return self.__empty_df__(dataframe)
        df_filter = dataframe[column_name].between(min_val, max_val, inclusive="both")
        return dataframe[df_filter].reset_index(drop=True)

    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Private filter used in run(). Requires configuration file."""
        print(self.params)
        numerical_range = self.params['range']
        return self.filter(dataframe, self.get_name(),
                           numerical_range["min"], numerical_range["max"])

    def _filter_empty(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Filter only the empty value: -1."""
        return self.filter(dataframe, self.params['colum_name'], -1, -1)


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
    return NumericalFilter(name, params, **kwargs)
