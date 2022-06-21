"""This module contains filtering constants that are used in other modules.

Constants:

    KEY_DATA_SUBSET: key that is used to identify a data subset.
    KEY_DATA_FILTER_PASS: key that is used to identify a data filter pass.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Dict, Any


FILTER_NUMERICAL = 'numerical'
FILTER_CATEGORICAL = 'categorical'
FILTER_COUNT = 'count'

KEY_DATA_SUBSET = 'subset'
KEY_DATA_FILTER_PASS = 'filter_pass'


def deduce_filter_type(params: Dict[str, Any]) -> str:
    """Get filter type ('numerical', 'categorical', 'count') from Configuration params.

    Args:
        params: Configuration parameters.

    Return:
        Either 'numerical', 'categorical' or 'count'. Default 'categorical'
    """
    keys = params.keys
    if 'min' in keys and 'max' in keys:
        return FILTER_NUMERICAL
    if 'values' in keys:
        return FILTER_CATEGORICAL
    if 'threshold' in keys:
        return FILTER_COUNT
    return FILTER_CATEGORICAL
