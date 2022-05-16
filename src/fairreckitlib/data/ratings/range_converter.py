"""This module contains the range converting functionality.

Classes:

    RangeConverter: can convert ratings with a scalar.

Functions:

    create_range_converter: create an instance of the class to be used in a factory.
    create_range_converter_params: create range convert parameters to be used in a factory.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from ...core.config_params import ConfigParameters
from ..set.dataset import DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT
from .base_converter import RatingConverter
from .convert_constants import RATING_TYPE_THRESHOLD


class RangeConverter(RatingConverter):
    """Range Converter on data ratings.

    Converts the rating column of the dataframe to a specified range.
    """

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Convert ratings in the dataframe.

        Takes the max value and divides all values so that
        they all fall within a range of [0,1], unless another upper
        bound is given by the parameters on creation. The rating can then
        also be multiplied by a scalar, e.g. when an implicit rating is needed.

        Args:
            dataframe: a df that should contain a 'rating' column.

        Returns:
            the converted dataframe and the type of rating, either 'explicit' or 'implicit'.
        """
        upper_bound = self.params['upper_bound']
        max_rating = dataframe.max()['rating']
        dataframe['rating'] = dataframe['rating'].apply(lambda x : x / max_rating * upper_bound)

        # any ratings above a RATING_TYPE_THRESHOLD will be viewed as implicit
        if upper_bound > RATING_TYPE_THRESHOLD:
            rating_type = DATASET_RATINGS_IMPLICIT
        else:
            rating_type = DATASET_RATINGS_EXPLICIT

        return dataframe, rating_type


def create_range_converter(name: str, params: Dict[str, Any]) -> RangeConverter:
    """Create the Range Converter.

    Args:
        name: the name of the converter.
        params: containing the following name-value pairs:
            upper_bound(float): the upper bound of the range restriction.

    Returns:
        the data range converter.
    """
    return RangeConverter(name, params)


def create_range_converter_params() -> ConfigParameters:
    """Create the parameters of the range converter.

    Returns:
        the configuration parameters of the converter.
    """
    params = ConfigParameters()
    params.add_value('upper_bound', float, 1.0, (1.0, 1000000.0))
    return params
