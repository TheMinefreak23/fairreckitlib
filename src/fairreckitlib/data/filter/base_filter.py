"""Module that provides a base for all three types of filters: Numerical, Categorical, Count.

Classes:

    DataFilter: Base filter class.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import pandas as pd

from ..set import dataset as ds
from ..data_modifier import DataModifier


class DataFilter(DataModifier, metaclass=ABCMeta):
    """Base class to filter a df (not a dataframe in particular).

    Public method:
        run
    """

    def __init__(self, name: str, params: Dict[str, Any], **kwargs):
        """Make Constructor of the class.

        Uses optional arguments to enable sole use of subclass.filter().

        Args:
            name: Configuration name of the filter.
            params: Configuration parameters.
        """
        DataModifier.__init__(self, name, params)
        self.dataset = kwargs['dataset']
        self.matrix_name = kwargs['matrix_name']

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Carry out the filtering.

        Args:
            dataframe: Dataframe to be filtered on.

        Return:
            The filtered dataframe.
        """
        # Filtering that requires external columns i.e., filter column not available in dataframe.
        return self._external_col_filter(dataframe)

    @abstractmethod
    def get_type(self) -> str:
        """Get the type of the filter.

        Returns:
            The type name of the filter.
        """
        raise NotImplementedError()

    @abstractmethod
    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Sugar coats subclasses' filter() for run and _external_col_filter as sugar.

        Raises:
            NotImplementedError: This method should be implemented in the subclasses.
        """
        raise NotImplementedError()

    def _external_col_filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """When filter needs a column from some dataset table located elsewhere.

        Args:
            dataframe: The dataframe to be filtered.

        Returns:
            A filtered dataframe.
        """
        # Add required columns
        og_cols = dataframe.columns
        new_dataframe = ds.add_dataset_columns(
            self.dataset, self.matrix_name, dataframe, [self.get_name()])
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
