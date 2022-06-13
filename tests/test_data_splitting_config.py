"""This module tests the formatting and parsing of the dataset splitting configuration.

Functions:

    test_parse_data_split_config: test parsing the data splitter configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.data.split.split_config import SplitConfig, create_default_split_config
from src.fairreckitlib.data.split.split_config_parsing import parse_data_split_config
from src.fairreckitlib.data.split.split_constants import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from src.fairreckitlib.data.split.split_constants import MIN_TEST_RATIO, MAX_TEST_RATIO
from src.fairreckitlib.data.split.split_factory import create_split_factory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry

INVALID_CONTAINER_TYPES = [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]


def test_parse_data_split_config(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing the data splitter configuration for all available datasets and splitters."""
    split_factory = create_split_factory()

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            # test failure for missing KEY_SPLITTING
            parsed_split_config = parse_data_split_config(
                {},
                dataset,
                matrix_name,
                split_factory,
                parse_event_dispatcher
            )
            assert parsed_split_config == create_default_split_config(), \
                'expected default splitting config when parsing empty split configuration'

            # test failure for parsing various types, including a dict that is empty
            for split_config in INVALID_CONTAINER_TYPES:
                parsed_split_config = parse_data_split_config(
                    {KEY_SPLITTING: split_config},
                    dataset,
                    matrix_name,
                    split_factory,
                    parse_event_dispatcher
                )
                assert parsed_split_config == create_default_split_config(), \
                    'expected default splitting config when parsing invalid split configuration'

            # test failure for unknown splitter name
            parsed_split_config = parse_data_split_config(
                {KEY_SPLITTING: {KEY_NAME: 'unknown'}},
                dataset,
                matrix_name,
                split_factory,
                parse_event_dispatcher
            )
            assert parsed_split_config == create_default_split_config(), \
                'expected default splitting config when parsing unknown split configuration'

            for splitter_name in split_factory.get_available_names():
                splitter_params = split_factory.create_params(splitter_name)
                split_config = {KEY_NAME: splitter_name}
                # only add params when there are any, to compare with the formatting later
                if splitter_params.get_num_params() > 0:
                    split_config[KEY_PARAMS] = splitter_params.get_defaults()

                for test_ratio in [MIN_TEST_RATIO, 0.2, 0.5, 0.8, MAX_TEST_RATIO]:
                    split_config[KEY_SPLIT_TEST_RATIO] = test_ratio

                    # test successfully parsing a test ratio
                    parsed_split_config = parse_data_split_config(
                        {KEY_SPLITTING: split_config},
                        dataset,
                        matrix_name,
                        split_factory,
                        parse_event_dispatcher
                    )
                    assert isinstance(parsed_split_config, SplitConfig), \
                        'expected SplitConfig to be parsed'
                    assert parsed_split_config.test_ratio == test_ratio, \
                        'expected input test ratio to be parsed successfully'
                    assert parsed_split_config.to_yml_format() == split_config, \
                        'expected parsed split config to be formatted correctly'

                # test clamping of minimum test ratio
                split_config[KEY_SPLIT_TEST_RATIO] = MIN_TEST_RATIO - 0.0001
                parsed_split_config = parse_data_split_config(
                    {KEY_SPLITTING: split_config},
                    dataset,
                    matrix_name,
                    split_factory,
                    parse_event_dispatcher
                )
                assert parsed_split_config.test_ratio == MIN_TEST_RATIO, \
                    'expected test ratio to be clamped to the minimum test ratio'

                # test clamping of maximum test ratio
                split_config[KEY_SPLIT_TEST_RATIO] = MAX_TEST_RATIO + 0.0001
                parsed_split_config = parse_data_split_config(
                    {KEY_SPLITTING: split_config},
                    dataset,
                    matrix_name,
                    split_factory,
                    parse_event_dispatcher
                )
                assert parsed_split_config.test_ratio == MAX_TEST_RATIO, \
                    'expected test ratio to be clamped to the maximum test ratio'
