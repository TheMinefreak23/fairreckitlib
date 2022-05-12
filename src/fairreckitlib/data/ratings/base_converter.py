"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Any

import pandas as pd

class RatingConverter(metaclass=ABCMeta):
    """Base class for FairRecKit rating converters.

    A converter is used to convert ratings of a dataframe.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        """Construct the Rating Converter.

        Args:
            name: the name of the converter.
            params: the converter parameters.
        """
        self.__name = name
        self.params = params

    def get_name(self) -> str:
        """Get the name of the converter.

        Returns:
            (str) the converter name.
        """
        return self.__name

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the converter.

        Returns:
            (dict) with the converter parameters.
        """
        return dict(self.params)

    @abstractmethod
    def run(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Run the converter on the specified dataframe.

        Args:
            dataframe: with at least the 'rating' column.

        Returns:
            dataframe: the converted dataframe.
            rating_type: the converted type of rating, either 'explicit' or 'implicit'.
        """
        raise NotImplementedError()
