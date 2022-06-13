"""This module tests the formatting and parsing of the dataset rating converter configuration.

Functions:

    test_parse_data_convert_config: test parsing the data rating converter configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.data.ratings.convert_config import ConvertConfig
from src.fairreckitlib.data.ratings.convert_config_parsing import parse_data_convert_config
from src.fairreckitlib.data.ratings.convert_constants import KEY_RATING_CONVERTER
from src.fairreckitlib.data.ratings.rating_converter_factory import create_rating_converter_factory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry

INVALID_CONTAINER_TYPES = [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]


def test_parse_data_convert_config(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing the data rating converter configuration for all available datasets."""
    rating_converter_factory = create_rating_converter_factory(data_registry)

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)
        dataset_converter_factory = rating_converter_factory.get_factory(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            matrix_converter_factory = dataset_converter_factory.get_factory(matrix_name)

            # test failure for missing KEY_RATING_CONVERTER
            parsed_convert_config = parse_data_convert_config(
                {},
                dataset,
                matrix_name,
                rating_converter_factory,
                parse_event_dispatcher
            )
            assert not bool(parsed_convert_config), \
                'expected None when parsing empty rating converter configuration'

            # test failure for parsing various types, including a dict that is empty
            for convert_config in INVALID_CONTAINER_TYPES:
                parsed_convert_config = parse_data_convert_config(
                    {KEY_RATING_CONVERTER: convert_config},
                    dataset,
                    matrix_name,
                    rating_converter_factory,
                    parse_event_dispatcher
                )
                assert not bool(parsed_convert_config), \
                    'expected None when parsing invalid rating converter configuration'

            # test failure for unknown rating converter name
            parsed_convert_config = parse_data_convert_config(
                {KEY_RATING_CONVERTER: {KEY_NAME: 'unknown'}},
                dataset,
                matrix_name,
                rating_converter_factory,
                parse_event_dispatcher
            )
            assert not bool(parsed_convert_config), \
                'expected None when parsing unknown rating converter configuration'

            for converter_name in matrix_converter_factory.get_available_names():
                converter_params = matrix_converter_factory.create_params(converter_name)
                convert_config = {
                    KEY_NAME: converter_name,
                    KEY_PARAMS: converter_params.get_defaults()
                }
                parsed_convert_config = parse_data_convert_config(
                    {KEY_RATING_CONVERTER: convert_config},
                    dataset,
                    matrix_name,
                    rating_converter_factory,
                    parse_event_dispatcher
                )
                assert bool(parsed_convert_config), \
                    'expected parsing to succeed for valid rating converter configuration'
                assert isinstance(parsed_convert_config, ConvertConfig), \
                    'expected ConvertConfig to be parsed'
                assert parsed_convert_config.to_yml_format() == convert_config, \
                    'expected parsed convert config to be formatted correctly'
