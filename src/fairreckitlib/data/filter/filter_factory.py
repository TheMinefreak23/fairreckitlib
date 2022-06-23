"""This module combines all three types of filters into a factory.

Functions:
    create_filter_factory: Creates a factory of three filter objects.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.config.config_factories import GroupFactory
from ..set.dataset import Dataset
from ..set.dataset_registry import DataRegistry
from ..data_modifier import DataModifierFactory, create_data_modifier_factory
from .filter_params import (
    create_params_categorical, create_params_numerical, create_params_count)
from .filter_constants import KEY_DATA_SUBSET
from .numerical_filter import create_numerical_filter
from .categorical_filter import create_categorical_filter
from .count_filter import create_count_filter
from .filter_constants import FILTER_COUNT


# NUMERICAL ['rating', 'timestamp']
# CATEGORICAL, ['user_occupation','artist_genres','movie_genres']
# COUNT ['user_country', 'artist_genres', 'movie_genres']

def create_filter_factory(data_registry: DataRegistry) -> GroupFactory:
    """Create the dataframe filter factory.

    Args:

        data_registry: the data registry with available datasets.

    Returns:
        the factory with all available filters.
    """
    def on_add_entries(matrix_factory: DataModifierFactory, dataset: Dataset) -> None:
        """Add the filters to the matrix factory.

        Args:
            matrix_factory: the factory to add the filters to.
            dataset: the dataset associated with the matrix factory.
        """
        matrix_name = matrix_factory.get_name()

        for table_name, table_columns in dataset.get_available_columns(matrix_name).items():
            table_age = table_name + '_age'
            if table_age in table_columns:
                matrix_factory.add_obj(
                    table_age,
                    create_numerical_filter,
                    create_params_numerical
                )

            table_country = table_name + '_country'
            if table_country in table_columns:
                matrix_factory.add_obj(
                    table_country,
                    create_categorical_filter,
                    create_params_categorical
                )
                matrix_factory.add_obj(
                    table_country + '_' + FILTER_COUNT,
                    create_count_filter,
                    create_params_count
                )

            table_gender = table_name + '_gender'
            if table_gender in table_columns:
                matrix_factory.add_obj(
                    table_gender,
                    create_categorical_filter,
                    create_params_categorical
                    )

    return create_data_modifier_factory(data_registry, KEY_DATA_SUBSET, on_add_entries)
