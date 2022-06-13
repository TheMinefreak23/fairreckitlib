"""This module tests the formatting and parsing of the data (matrix) configurations.

Functions:

    test_parse_data_config: test parsing the data configuration from the experiment configuration.
    test_parse_data_matrix_config: test parsing the data matrix configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.data.ratings.convert_constants import KEY_RATING_CONVERTER
from src.fairreckitlib.data.data_factory import KEY_DATA, create_data_factory
from src.fairreckitlib.data.filter.filter_constants import KEY_DATA_SUBSET
from src.fairreckitlib.data.pipeline.data_config import DataMatrixConfig
from src.fairreckitlib.data.pipeline.data_config_parsing import \
    parse_data_config, parse_data_matrix_config
from src.fairreckitlib.data.set.dataset_constants import KEY_DATASET, KEY_MATRIX
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.data.split.split_config import create_default_split_config
from src.fairreckitlib.data.split.split_constants import KEY_SPLITTING

INVALID_CONTAINER_TYPES = [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]


def test_parse_data_config(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing the data configuration from the experiment configuration."""
    data_factory = create_data_factory(data_registry)

    parsed_data_config = parse_data_config(
        {},
        data_registry,
        data_factory,
        parse_event_dispatcher
    )
    assert not bool(parsed_data_config), \
        'expected None when parsing invalid data configuration'

    # test failure for parsing various types, including a list that is empty
    for data_matrices_config in INVALID_CONTAINER_TYPES:
        parsed_data_config = parse_data_config(
            {KEY_DATA: data_matrices_config},
            data_registry,
            data_factory,
            parse_event_dispatcher
        )
        assert not bool(parsed_data_config), \
            'expected None when parsing invalid data matrices configuration'

    parsed_data_config = parse_data_config(
        {KEY_DATA: [{KEY_DATASET: 'unknown'}]},
        data_registry,
        data_factory,
        parse_event_dispatcher
    )
    assert not bool(parsed_data_config), \
        'expected None when parsing data configuration with an unknown data matrix'

    data_matrices = [{
        KEY_DATASET: dataset_name,
        KEY_MATRIX: matrix_name,
        KEY_SPLITTING: create_default_split_config()
    } for dataset_name in data_registry.get_available_sets()
    for matrix_name in data_registry.get_set(dataset_name).get_available_matrices()]

    parsed_data_config = parse_data_config(
        {KEY_DATA: data_matrices},
        data_registry,
        data_factory,
        parse_event_dispatcher
    )
    assert bool(parsed_data_config), \
        'expected data config parsing to succeed for valid data matrices'
    assert isinstance(parsed_data_config, list), \
        'expected list of data matrix configurations to be parsed'
    assert len(parsed_data_config) == len(data_matrices), \
        'expected all data matrices to be parsed'


def test_parse_data_matrix_config(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing the data matrix configuration for each dataset-matrix pair."""
    data_factory = create_data_factory(data_registry)

    # test failure for parsing various types, including a dict that is empty
    for data_matrix_config in INVALID_CONTAINER_TYPES:
        parsed_data_matrix_config, parsed_data_matrix_name = parse_data_matrix_config(
            data_matrix_config,
            data_registry,
            data_factory,
            parse_event_dispatcher
        )
        assert not bool(parsed_data_matrix_config), \
            'expected None when parsing invalid data matrix configuration'
        assert not bool(parsed_data_matrix_name), \
            'did not expect dataset name to be parsed'

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            parsed_data_matrix_config, parsed_data_matrix_name = parse_data_matrix_config(
                {KEY_DATASET: dataset_name, KEY_MATRIX: matrix_name},
                data_registry,
                data_factory,
                parse_event_dispatcher
            )
            assert bool(parsed_data_matrix_config), \
                'expected data matrix parsing to succeed for a known dataset-matrix pair'
            assert isinstance(parsed_data_matrix_config, DataMatrixConfig), \
                'expected DataMatrixConfig to be parsed'
            assert not bool(parsed_data_matrix_config.converter), \
                'did not expect rating converter to be parsed'
            assert parsed_data_matrix_config.splitting == create_default_split_config(), \
                'expected default splitting to be parsed'

            formatted_data_matrix_config = parsed_data_matrix_config.to_yml_format()
            assert KEY_DATA_SUBSET not in formatted_data_matrix_config, \
                'did not expect any prefilters to be present in the formatted configuration'
            assert KEY_RATING_CONVERTER not in formatted_data_matrix_config, \
                'did not expect rating converter to be present in the formatted configuration'
            assert KEY_SPLITTING in formatted_data_matrix_config, \
                'expected splitting to be present in the formatted configuration'
            assert KEY_DATASET in formatted_data_matrix_config, \
                'expected dataset name to be present in the formatted configuration'
            assert KEY_MATRIX in formatted_data_matrix_config, \
                'expected dataset matrix name to be present in the formatted configuration'
