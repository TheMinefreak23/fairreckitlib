"""This module contains functionality to create a converter factory.

Functions:

    create_rating_converter_factory: create a factory with converters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.factories import Factory, create_factory_from_list
from .convert_constants import KEY_RATING_CONVERTER, CONVERTER_KL, CONVERTER_RANGE
from .range_converter import create_range_converter, create_range_converter_params
from .kl_converter import create_kl_converter, create_kl_converter_params


def create_rating_converter_factory() -> Factory:
    """Create the rating converter factory.

    Returns:
        the factory with all available converters.
    """
    return create_factory_from_list(KEY_RATING_CONVERTER, [
        (CONVERTER_KL,
         create_kl_converter,
         create_kl_converter_params
         ),
        (CONVERTER_RANGE,
         create_range_converter,
         create_range_converter_params
         )
    ])
