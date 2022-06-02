"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict
import pandas as pd
from .base_filter import DataFilter


class CountFilter(DataFilter):
    """Filter the dataframe on categorical data,
    and selects only those of which its count is above a given threshold.

    Public method:
        filter
    """

    def filter(self, dataframe: pd.DataFrame, column_name='', threshold: int=1) -> pd.DataFrame:
        """Filter out the values in column_name which count is below threshold.

        Args:
            dataframe: Dataframe to be filtered.
            column_name (str): Name of the column.
            threshold (int):
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
        return self.filter(dataframe, self.params['column_name'], self.params['threshold'])


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
