"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import pandas as pd
from fairreckitlib.data.set.dataset import Dataset

import fairreckitlib.data.set.dataset as ds

class DataFilter(metaclass=ABCMeta):
    """Base class to filter a df (not a dataframe in particular), as long as the df
    contains a 'user' and 'item' column.

    Together with a factory pattern similar to the data.split module
    we can define a variety of filters to exclude rows that do not satisfy the filter
    from the specified df as long as it retains the 'user' and 'item' columns.

    These filters could be used by the fair-rec-kit-app as well (table browsing).

    Public method:
        run
    """

    def __init__(self, name='', params=None, **kwargs) -> None:
        """Make Constructor of the class.

        Uses optional arguments to enable sole use of subclass.filter().

        Args:
            name (str): Configuration name of the filter.
            params (Dict[str, Any]): Configuration parameters.

        """
        self.name = name
        self.params = params
        # self.column_name = params['column_name']  # not needed but! needs verification..

    def run(self, dataframe: pd.DataFrame, _dataset: Dataset=None,
            matrix_name: str='') -> pd.DataFrame:
        """Carry out the filtering.

        Args:
            dataframe: Dataframe to be filtered on.
            _dataset (Optional): Dataset object the external column is retrieved from.
            matrix_name (Optional): Name of the matrix inside the dataset.

        Return:
            The filtered dataframe.
        """
        # Filtering that requires external columns i.e., filter column not available in dataframe.
        if _dataset and matrix_name:
            return self._external_col_filter(dataframe, _dataset, matrix_name)
        # Filter using dataframe that is assumed to be complete.
        return self._filter(dataframe)

    @abstractmethod
    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Basic filter functionality used in the data pipeline.

        Raises:
            NotImplementedError: This method should be implemented in the subclasses.
        """
        raise NotImplementedError()

    def _external_col_filter(self, dataframe: pd.DataFrame, _dataset: Dataset,
        matrix_name: str) -> pd.DataFrame:
        """When filter needs a column from some dataset table located elsewhere.

        Args:
            dataframe: The dataframe to be filtered.

        Returns:
            A filtered dataframe.
        """
        # Add required columns
        og_cols = dataframe.columns
        new_dataframe = ds.add_dataset_columns(
            _dataset, matrix_name, dataframe, [self.params['column_name']])
        new_cols = new_dataframe.columns

        new_dataframe = self._filter(new_dataframe)

        # Remove columns not in original dataframe
        for new_col in new_cols:
            if new_col not in og_cols:
                new_dataframe = new_dataframe.drop([new_col], axis=1, errors='ignore')
        return new_dataframe

    @staticmethod
    def __empty_df__(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return an empty dataframe with same columns."""
        return dataframe.iloc[:0,:].copy()

    def __str__(self):
        """To string.

        Returns:
            The name of the class.
        """
        return self.__class__.__name__
