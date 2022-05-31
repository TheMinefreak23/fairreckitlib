"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List
import pandas as pd
from .base_filter import DataFilter

class CategoricalFilter(DataFilter):
    """Filters the dataframe on categorical data, such as country or gender.
    
    Public method:
        filter
    """

    def filter(self, dataframe: pd.DataFrame, column_name='', conditions=[]) -> pd.DataFrame:
        """Filters on a list of categories.

        Args:
            dataframe: Dataframe to be filtered.
            column_name (str): Name of the column where the conditions need to be met.
            conditions (List[Any]): A list of values, where values of the column_name in the resulting dataframe meet some condition.
            
        Returns:
            A filtered dataframe.
        """
        if column_name not in dataframe.columns:
            return dataframe
        df_filter = dataframe[column_name].isin(conditions)
        return dataframe[df_filter].reset_index(drop=True)

    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Private filter used in run(). Requires configuration file."""
        return self.filter(dataframe, self.params['name'], self.params['values'])



def create_categorical_filter(name: str, params: Dict[str, Any], **kwargs) -> DataFilter:
    """Create an instance of the class CategoricalFilter

    Args:
        name: Name of the filter.
        params: Configuration file.
        **kwargs: Contains dataset and matrix_name.

    Returns:
        An instance of the CategoricalFilter class.
    """
    return CategoricalFilter(name, params, kwargs)
