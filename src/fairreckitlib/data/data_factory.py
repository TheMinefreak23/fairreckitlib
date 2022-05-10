"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.factories import GroupFactory
from .pipeline.data_config import KEY_DATASETS
from .ratings.rating_modifier_factory import create_rating_modifier_factory
from .split.split_factory import create_split_factory


def create_data_factory():
    data_factory = GroupFactory(KEY_DATASETS)
    data_factory.add_factory(create_rating_modifier_factory())
    data_factory.add_factory(create_split_factory())
    return data_factory
