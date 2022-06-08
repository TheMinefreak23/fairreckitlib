"""This module tests the data (modifier) factories.

Functions:

    test_data_factory: test data factories to be derived from the correct base class.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.config.config_factories import Factory, GroupFactory
from src.fairreckitlib.data.data_factory import create_data_factory
from src.fairreckitlib.data.filter.filter_constants import KEY_DATA_FILTERS
from src.fairreckitlib.data.ratings.convert_constants import KEY_RATING_CONVERTER
from src.fairreckitlib.data.split.split_constants import KEY_SPLITTING
from src.fairreckitlib.data.set.dataset_registry import DataRegistry


def test_data_factory(data_registry: DataRegistry) -> None:
    """Test factories in the factory to be derived from the correct base class."""
    data_factory = create_data_factory(data_registry)
    assert isinstance(data_factory, GroupFactory), 'expected data modifier group factory.'

    assert bool(data_factory.get_factory(KEY_DATA_FILTERS)), 'missing data filters factory.'
    assert bool(data_factory.get_factory(KEY_RATING_CONVERTER)),'missing rating converter factory.'
    assert bool(data_factory.get_factory(KEY_SPLITTING)),'missing data splitter factory.'

    assert isinstance(data_factory.get_factory(KEY_DATA_FILTERS), GroupFactory), \
        'expected dataset filter group factory'
    assert isinstance(data_factory.get_factory(KEY_RATING_CONVERTER), GroupFactory), \
        'expected dataset rating converter group factory'
    assert isinstance(data_factory.get_factory(KEY_SPLITTING), Factory), \
        'expected dataset splitting factory'
