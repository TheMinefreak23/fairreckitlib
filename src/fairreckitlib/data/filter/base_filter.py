"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict
import pandas as pd

from fairreckitlib.data.set import dataset


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
        self.kwargs = kwargs

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Carry out the filtering."""
        return self._external_col_filter(dataframe)

    @abstractmethod
    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Basic filter functionality used in the data pipeline.

        Raises:
            NotImplementedError: This method should be implemented in the subclasses.
        """
        raise NotImplementedError()

    def _external_col_filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """When filter needs a column from some dataset table located elsewhere.
        
        Args:
            dataframe: The dataset to be filtered.
        
        Returns:
            A filtered dataframe.
        """ 
        #  kwargs

        # Add required columns
        og_cols = dataframe.columns()
        new_dataframe = dataset.add_dataset_columns(self.kwargs['dataset'], self.kwargs['matrix_name'], dataframe, [self.params['column_name']])
        new_cols = new_dataframe.columns()
        
        self._filter(dataframe)

        # Remove columns not in original dataframe
        for i in range(len(new_cols)):
            if new_cols[i] not in og_cols:
                new_dataframe.drop([new_cols[i]], axis=1, error='ignore')
        return new_dataframe

    def __str__(self):
        """To string.

        Returns:
            The name of the class.
        """
        return self.__class__.__name__
