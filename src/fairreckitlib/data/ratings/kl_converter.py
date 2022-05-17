"""This module contains the Kullback-Leibler converter.

Classes:

    KLConverter: can convert ratings using the Kullback-Leibler formula.

Functions:

    create_kl_converter: create an instance of the class (factory creation compatible).
    create_kl_converter_params: create kl converter config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from ...core.config_params import ConfigParameters
from .base_converter import RatingConverter


class KLConverter(RatingConverter):
    """Kullback-Leibler Converter on data ratings.

    Applies the Kullback-Leibler formula to the rating column of the dataframe.
    """

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Apply the Kullback-Leibler formula to convert ratings.

        Args:
            dataframe: with 'user', 'item' and 'rating' columns.

        Returns:
            the converted dataframe and the type of rating, either 'explicit' or 'implicit'.
        """
        # method = self.params['method']
        # TODO apply kullback-leibler formula on the rating column. Needs APC/ALC arg.
        raise NotImplementedError()


def create_kl_converter(name: str, params: Dict[str, Any]) -> KLConverter:
    """Create the KL Converter.

    Args:
        name: the name of the converter.
        params: containing the following name-value pairs:
            method(str): the method to apply, either 'APC' or 'ALC'.

    Returns:
        the data kl converter.
    """
    return KLConverter(name, params)


def create_kl_converter_params() -> ConfigParameters:
    """Create the parameters of the kl converter.

    Returns:
        the configuration parameters of the converter.
    """
    methods = ['APC', 'ALC']

    params = ConfigParameters()
    params.add_option('method', str, methods[0], methods)
    return params
