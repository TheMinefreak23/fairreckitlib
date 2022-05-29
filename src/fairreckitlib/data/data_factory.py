"""This module contains functionality to create a data factory.

Constants:

    KEY_DATA: key that is used to identify data.

Functions:

    create_data_factory: create factory with data modifier factories.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.factories import GroupFactory
from .ratings.rating_converter_factory import create_rating_converter_factory
from .split.split_factory import create_split_factory

KEY_DATA = 'data'


def create_data_factory() -> GroupFactory:
    """Create a group factory with all data modifiers.

    Consists of two data modifier factories:
        1) data rating converters.
        2) data splitters.

    Returns:
        the group factory with available data modifier factories.
    """
    data_factory = GroupFactory(KEY_DATA)
    data_factory.add_factory(create_rating_converter_factory())
    data_factory.add_factory(create_split_factory())
    return data_factory
