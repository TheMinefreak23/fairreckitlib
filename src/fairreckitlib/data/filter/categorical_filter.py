"""Module to filter on categories, like country or gender.

Classes:

    CategoricalFilter: Filter the dataframe on categorical data, such as country or gender.

Functions:

    create_categorical_filter: Create an instance of CategoricalFilter.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List
import numpy
import pandas as pd
from .base_filter import DataFilter


class CategoricalFilter(DataFilter):
    """Filter the dataframe on categorical data, such as country or gender.

    Public method:
        filter
    """

    def filter(self, dataframe: pd.DataFrame, column_name='',
               conditions: List[Any]=None) -> pd.DataFrame:
        """Filter on a list of categories.

        Args:
            dataframe: Dataframe to be filtered.
            column_name (str): Name of the column where the conditions need to be met.
            conditions (List[Any]): A list of values,
                where values of the column_name in the resulting dataframe meet some condition.

        Returns:
            A filtered dataframe.
        """
        if column_name not in dataframe.columns:
            return self.__empty_df__(dataframe)
        self._handle_none_value(conditions)
        df_filter = dataframe[column_name].isin(conditions)
        return dataframe[df_filter].reset_index(drop=True)

    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Private filter used in run(). Requires configuration file."""
        return self.filter(dataframe, self.params['column_name'], self.params['values'])

    @staticmethod
    def _handle_none_value(conditions: List[Any]):
        """Change None value to empty value: numpy.NaN."""
        if conditions is None:
            conditions = []
        if None in conditions:
            conditions.remove(None)
            conditions.append(numpy.NaN)


def create_categorical_filter(name: str, params: Dict[str, Any], **kwargs) -> DataFilter:
    """Create an instance of the class CategoricalFilter.

    Args:
        name: Name of the filter.
        params: Configuration file.
        **kwargs: Contains dataset and matrix_name.

    Returns:
        An instance of the CategoricalFilter class.
    """
    return CategoricalFilter(name, params, **kwargs)
