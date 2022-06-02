"""Module to filter on count threshold. Can be used to filter on countries with many users.

Classes:

    CountFilter: Filter the dataframe on a column, such as country.
        Show only those above a certain threshold.

Functions:

    create_count_filter: Create an instance of CountFilter.


This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict
import pandas as pd
from .base_filter import DataFilter
from .filter_constants import FILTER_COUNT

class CountFilter(DataFilter):
    """Filter the dataframe on a column, and select only whose count is above a given threshold.

    Public method:
        filter
    """

    def get_type(self) -> str:
        """Get the type of the filter.

        Returns:
            The type name of the filter.
        """
        return FILTER_COUNT

    def filter(self, dataframe: pd.DataFrame,
               column_name: str='', threshold: int=1) -> pd.DataFrame:
        """Filter out the values in column_name which count is below threshold.

        Args:
            dataframe: Dataframe to be filtered.
            column_name: Name of the column.
            threshold:
                Values above or equal to the threshold will be included in the resulting dataframe.

        Returns:
            A filtered dataframe.
        """
        if column_name not in dataframe.columns:
            return self.__empty_df__(dataframe)
        value_counts = dataframe[column_name].value_counts()
        key_dict = (value_counts >= threshold).to_dict()
        df_filter = dataframe[column_name].map(key_dict)
        return dataframe[df_filter].reset_index(drop=True)

    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Private filter used in run(). Requires configuration file."""
        return self.filter(dataframe, self.get_name().removesuffix('_' + FILTER_COUNT),
                           self.params['threshold'])


def create_count_filter(name: str, params: Dict[str, Any], **kwargs) -> CountFilter:
    """Create an instance of the class CountFilter.

    Args:
        name: Name of the filter.
        params: Configuration file.
        **kwargs: Contains dataset and matrix_name.

    Returns:
        An instance of the CountFilter class.
    """
    return CountFilter(name, params, **kwargs)
