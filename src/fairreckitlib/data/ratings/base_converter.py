"""This module contains the base class for converting ratings.

CLasses:

    RatingConverter: the base class for converting ratings.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from ..data_modifier import DataModifier


class RatingConverter(DataModifier):
    """Base class for FairRecKit rating converters.

    A converter is used to convert ratings of a dataframe.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        """Construct the Rating Converter.

        Args:
            name: the name of the converter.
            params: the converter parameters.
        """
        DataModifier.__init__(self, name, params)

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Run the converter on the specified dataframe.

        Args:
            dataframe: with at least the 'rating' column.

        Returns:
            the converted dataframe and the type of rating, either 'explicit' or 'implicit'.
        """
        raise NotImplementedError()
