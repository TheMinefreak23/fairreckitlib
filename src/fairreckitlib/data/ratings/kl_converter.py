"""This module contains the Kullback-Leibler converter.

This way of converting is not implemented in the data pipeline.
The intended use of this stems from the following paper about mainstreaminess:

https://www.christinebauer.eu/publications/bauer-2019-plosone-mainstreaminess/

see pages 10-11.

This paper describes an altered version of the Kullback-Leibler formla
and converts implicit ratings to explicit ratings in the range [0,1].

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

from ...core.params.config_parameters import ConfigParameters
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
        # TODO apply kullback-leibler formula on the rating column. Needs APC/ALC arg.
        # method = self.params['method']
        # use the APC/ALC parameter, import count.py from this package to calculate either
        # apply the altered kl formula to each row of the given dataframe:
        #       1 / np.mean(1 - np.exp(-KL(P||Q)), 1 - np.exp(-KL(Q||P)))
        #       where
        #           KL(P||Q) = sum(A.C(u) * np.log(A.C(u) / A.C   ))
        #           KL(Q||P) = sum(A.C    * np.log(A.C    / A.C(u)))
        #   note that the above is a slightly abstract and incomplete notation,
        #   please consult the paper linked at the top.
        # return (converted_dataframe, 'explicit')
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
    params.add_single_option('method', str, methods[0], methods)
    return params
