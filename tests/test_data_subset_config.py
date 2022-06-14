"""This module tests the formatting and parsing of the data subset configurations.

Functions:

    test_parse_data_subset_config: test parsing data subset configuration.
    test_parse_data_filter_passes: test parsing multiple filter pass configurations.
    test_parse_data_filter_pass_config: test parsing data filter pass configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.config.config_yml import format_yml_config_list
from src.fairreckitlib.core.core_constants import KEY_NAME
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.data.filter.filter_config import DataSubsetConfig, FilterPassConfig
from src.fairreckitlib.data.filter.filter_config_parsing import \
    parse_data_subset_config, parse_data_filter_passes, parse_data_filter_pass_config
from src.fairreckitlib.data.filter.filter_constants import KEY_DATA_FILTER_PASS, KEY_DATA_SUBSET
from src.fairreckitlib.data.filter.filter_factory import create_filter_factory
from src.fairreckitlib.data.set.dataset_constants import KEY_DATASET, KEY_MATRIX
from src.fairreckitlib.data.set.dataset_registry import DataRegistry

INVALID_CONTAINER_TYPES = \
    [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]


def test_parse_data_subset_config(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing data subset configuration for each dataset-matrix pair."""
    data_filter_factory = create_filter_factory(data_registry)

    # test failure for unknown dataset
    parsed_data_subset_config, parsed_data_matrix_name = parse_data_subset_config(
        {KEY_DATASET: 'unknown'},
        data_registry,
        data_filter_factory,
        parse_event_dispatcher
    )
    assert not bool(parsed_data_subset_config), \
        'expected None when parsing unknown dataset from the configuration'
    assert parsed_data_matrix_name, \
        'expected dataset name to already be parsed'

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)
        dataset_filter_factory = data_filter_factory.get_factory(dataset_name)

        # test failure for unknown matrix
        parsed_data_subset_config, parsed_data_matrix_name = parse_data_subset_config(
            {KEY_DATASET: dataset_name, KEY_MATRIX: 'unknown'},
            data_registry,
            data_filter_factory,
            parse_event_dispatcher
        )
        assert not bool(parsed_data_subset_config), \
            'expected None when parsing unknown dataset matrix from the configuration'
        assert dataset_name in parsed_data_matrix_name, \
            'expected dataset name to already be parsed'

        for matrix_name in dataset.get_available_matrices():
            # test success without the optional subset configuration
            parsed_data_subset_config, parsed_data_matrix_name = parse_data_subset_config(
                {KEY_DATASET: dataset_name, KEY_MATRIX: matrix_name},
                data_registry,
                data_filter_factory,
                parse_event_dispatcher
            )
            assert bool(parsed_data_subset_config), \
                'expected data matrix parsing to succeed for a known dataset-matrix pair'
            assert isinstance(parsed_data_subset_config, DataSubsetConfig), \
                'expected DataSubsetConfig to be parsed'
            assert dataset_name in parsed_data_matrix_name, \
                'expected dataset name to already be parsed'
            assert matrix_name in parsed_data_matrix_name, \
                'expected matrix name to already be parsed'
            assert len(parsed_data_subset_config.filter_passes) == 0, \
                'did not expect any filter_passes to be parsed'

            formatted_data_matrix_config = parsed_data_subset_config.to_yml_format()
            assert KEY_DATA_SUBSET not in formatted_data_matrix_config, \
                'did not expect any filter_passes to be present in the formatted configuration'
            assert KEY_DATASET in formatted_data_matrix_config, \
                'expected dataset name to be present in the formatted configuration'
            assert KEY_MATRIX in formatted_data_matrix_config, \
                'expected dataset matrix name to be present in the formatted configuration'

            matrix_filter_factory = dataset_filter_factory.get_factory(matrix_name)

            # test success with the optional subset configuration
            subset = [{
                KEY_DATA_FILTER_PASS: [{KEY_NAME: name}]
            } for name in matrix_filter_factory.get_available_names()]

            parsed_data_subset_config, _ = parse_data_subset_config(
                {KEY_DATASET: dataset_name, KEY_MATRIX: matrix_name, KEY_DATA_SUBSET: subset},
                data_registry,
                data_filter_factory,
                parse_event_dispatcher
            )
            assert bool(parsed_data_subset_config)


def test_parse_data_filter_passes(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing data filter passes for each dataset-matrix pair."""
    data_filter_factory = create_filter_factory(data_registry)
    # test failure for empty dictionary without KEY_DATA_SUBSET
    assert not parse_data_filter_passes(
        'test',
        {},
        (None, 'not used'),
        data_filter_factory,
        parse_event_dispatcher
    )

    for filter_passes in INVALID_CONTAINER_TYPES:
        # test failure for parsing various types, including a list that is empty
        assert not parse_data_filter_passes(
            'test',
            {KEY_DATA_SUBSET: filter_passes},
            (None, 'not used'),
            data_filter_factory,
            parse_event_dispatcher
        )

        # test failure for a list with various invalid values
        assert not parse_data_filter_passes(
            'test',
            {KEY_DATA_SUBSET: [filter_passes]},
            (None, 'not used'),
            data_filter_factory,
            parse_event_dispatcher
        )

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)
        dataset_filter_factory = data_filter_factory.get_factory(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            matrix_filter_factory = dataset_filter_factory.get_factory(matrix_name)

            # generate filter passes for each available filter separately
            filter_passes = [{
                KEY_DATA_FILTER_PASS: [{KEY_NAME: name}]
            } for name in matrix_filter_factory.get_available_names()]

            # test success
            parsed_filter_passes = parse_data_filter_passes(
                'test',
                {KEY_DATA_SUBSET: filter_passes},
                (dataset, matrix_name),
                data_filter_factory,
                parse_event_dispatcher
            )
            assert isinstance(parsed_filter_passes, list), 'expected filter list to be parsed'
            formatted_filter_passes = format_yml_config_list(parsed_filter_passes)

            assert len(formatted_filter_passes) == len(filter_passes), \
                'expected all filter passes to be formatted from the original filter_passes'
            assert len(parsed_filter_passes) == len(filter_passes), \
                'expected all filter passes to be parsed from the original filter_passes'


def test_parse_data_filter_pass_config(
        data_registry: DataRegistry, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing data filter pass configuration for all filters of each dataset-matrix pair."""
    data_filter_factory = create_filter_factory(data_registry)

    # test failure for parsing various types, including a dict that is empty
    for filter_pass_config in INVALID_CONTAINER_TYPES:
        parsed_filter_pass = parse_data_filter_pass_config(
            'test',
            filter_pass_config,
            (None, 'not used'),
            GroupFactory('not used'),
            parse_event_dispatcher
        )
        assert not bool(parsed_filter_pass), \
            'did not expect parsed filter pass for invalid configuration'

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)
        dataset_filter_factory = data_filter_factory.get_factory(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            matrix_filter_factory = dataset_filter_factory.get_factory(matrix_name)

            # test failure for empty list of filters
            parsed_filter_pass = parse_data_filter_pass_config(
                'test',
                {KEY_DATA_FILTER_PASS: []},
                (dataset, matrix_name),
                data_filter_factory,
                parse_event_dispatcher
            )
            assert not bool(parsed_filter_pass), \
                'did not expect parsed filter pass that has no filters specified'

            # generate filter pass config with all available filters
            filter_pass_config = {KEY_DATA_FILTER_PASS: [
                    {KEY_NAME: name} for name in matrix_filter_factory.get_available_names()
                ]}
            num_filters = len(filter_pass_config[KEY_DATA_FILTER_PASS])
            # skip if no filters are generated
            if num_filters == 0:
                continue

            # test success for all available filters, including the formatted parse result
            parsed_filter_pass = parse_data_filter_pass_config(
                'test',
                filter_pass_config,
                (dataset, matrix_name),
                data_filter_factory,
                parse_event_dispatcher
            )
            formatted_filter_list = parsed_filter_pass.to_yml_format()[KEY_DATA_FILTER_PASS]

            assert bool(parsed_filter_pass), \
                'expected filter pass with all available filters to be parsed successfully'
            assert isinstance(parsed_filter_pass, FilterPassConfig), \
                'expected FilterPassConfig to be parsed'
            assert len(formatted_filter_list) == num_filters, \
                'expected all filters to be formatted from the filter pass configuration'
            assert len(parsed_filter_pass.filters) == num_filters, \
                'expected all filters to be parsed from the original filter pass configuration'

            for filter_name in matrix_filter_factory.get_available_names():
                assert any(filter_name is f[KEY_NAME] for f in formatted_filter_list), \
                    'expected available filter to be in the formatted filter pass'
                assert any(f.name == filter_name for f in parsed_filter_pass.filters), \
                    'expected available filter to be in the parsed filter pass'
