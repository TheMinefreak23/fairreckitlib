"""This module contains functionality to create the rating converter factory.

Functions:

    create_rating_converter_factory: create a factory with rating converters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.config.config_factories import GroupFactory
from ..data_modifier import DataModifierFactory
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
    factory = GroupFactory(KEY_RATING_CONVERTER)

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)
        dataset_factory = GroupFactory(dataset.get_name())

        factory.add_factory(dataset_factory)

        for matrix_name in dataset.get_available_matrices():
            matrix_factory = DataModifierFactory(matrix_name, dataset)
            matrix_factory.add_obj(CONVERTER_KL, create_kl_converter, create_kl_converter_params)
            matrix_factory.add_obj(CONVERTER_RANGE, create_range_converter, create_range_converter_params)

            dataset_factory.add_factory(matrix_factory)

    return factory
