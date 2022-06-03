"""This module contains functionality to create the rating converter factory.

Functions:

    create_rating_converter_factory: create a factory with rating converters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.config.config_factories import GroupFactory
from ..data_modifier import DataModifierFactory, create_data_modifier_factory
from ..set.dataset_registry import DataRegistry
from .convert_constants import KEY_RATING_CONVERTER, CONVERTER_KL, CONVERTER_RANGE
from .range_converter import create_range_converter, create_range_converter_params
from .kl_converter import create_kl_converter, create_kl_converter_params


def create_rating_converter_factory(data_registry: DataRegistry) -> GroupFactory:
    """Create the rating converter factory.

    Args:

        data_registry: the data registry with available datasets.

    Returns:
        the factory with all available converters.
    """
    def on_add_entries(matrix_factory: DataModifierFactory, _) -> None:
        """Add the rating converters to the matrix factory.

        Args:
            matrix_factory: the factory to add the converters to.
            _: the dataset associated with the matrix factory.

        """
        # add kl converter
        matrix_factory.add_obj(
            CONVERTER_KL,
            create_kl_converter,
            create_kl_converter_params
        )
        # add range converter
        matrix_factory.add_obj(
            CONVERTER_RANGE,
            create_range_converter,
            create_range_converter_params
        )

    return create_data_modifier_factory(data_registry, KEY_RATING_CONVERTER, on_add_entries)
