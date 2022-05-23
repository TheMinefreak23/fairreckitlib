"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict
import pandas as pd


class DataFilter(metaclass=ABCMeta):
    """Base class to filter a df (not a dataframe in particular), as long as the df
    contains a 'user' and 'item' column.

    Dataset class is not really necessary here, but when combined with the
    data.format module it needs to know about them to construct them.

    Together with a factory pattern similar to the data.split module
    we can define a variety of filters to exclude rows that do not satisfy the filter
    from the specified df as long as it retains the 'user' and 'item' columns.

    These filters could be used by the fair-rec-kit-app as well (table browsing).

    Public method:
        run
    """

    def __init__(self, name: str, params: Dict[str, Any]) -> None:
        """Make Constructor of the class."""
        self.name = name
        self.params = params

    @abstractmethod
    def run(self, dataframe: pd.DataFrame):
        """Carry out the filtering

        Raises:
            NotImplementedError: this method should be implimented in the subclasses
        """
        raise NotImplementedError()

    def __str__(self):
        """To string

        Returns:
            name of the class
        """
        return self.__class__.__name__
