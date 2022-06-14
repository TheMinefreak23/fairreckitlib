"""This module contains parsers for the data subgroup/filter configurations.

Functions:

    parse_data_subset_config: parse data subset configuration with multiple filter passes.
    parse_data_filter_passes: parse multiple filter pass configurations.
    parse_data_filter_pass_config: parse data filter pass configuration to multiple filters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_assert import \
    assert_is_container_not_empty, assert_is_key_in_dict, assert_is_one_of_list, assert_is_type
from ...core.parsing.parse_config_object import parse_config_object_list
from ..set.dataset import Dataset
from ..set.dataset_constants import KEY_DATASET, KEY_MATRIX
from ..set.dataset_registry import DataRegistry
from .filter_constants import KEY_DATA_FILTER_PASS, KEY_DATA_SUBSET
from .filter_config import DataSubsetConfig, FilterPassConfig, FilterConfig


def parse_data_subset_config(
        data_subset_config: Dict[str, Any],
        data_registry: DataRegistry,
        data_filter_factory: GroupFactory,
        event_dispatcher: EventDispatcher,
        *,
        data_parent_name: str=None,
        required: bool=True) -> Union[Tuple[DataSubsetConfig, str], Tuple[None, None]]:
    """Parse a data subset configuration.

    Args:
        data_subset_config: the data subset configuration.
        data_registry: the data registry containing the available datasets.
        data_filter_factory: factory with available dataset-matrix filter factories.
        event_dispatcher: to dispatch the parse event on failure.
        data_parent_name: the data parent name related to the data subset.
        required: whether parsing the subset is required to succeed.

    Returns:
        parsed_config: the parsed configuration or None on failure.
        dataset_name: the name of the parsed dataset or None on failure.
    """
    # assert dataset name is present
    if not assert_is_key_in_dict(
        KEY_DATASET,
        data_subset_config,
        event_dispatcher,
        'PARSE ERROR: missing key \'' + KEY_DATASET + '\' (required)' if required else ''
    ): return None, None

    dataset_name = data_subset_config[KEY_DATASET]
    parse_err = 'PARSE ERROR: ' + (data_parent_name + ' ' if data_parent_name else '')

    # assert dataset name is available in the data registry
    if not assert_is_one_of_list(
        dataset_name,
        data_registry.get_available_sets(),
        event_dispatcher,
        parse_err + 'unknown dataset name \'' + str(dataset_name) + '\''
    ): return None, dataset_name

    dataset = data_registry.get_set(dataset_name)

    # assert dataset matrix name is present
    if not assert_is_key_in_dict(
        KEY_MATRIX,
        data_subset_config,
        event_dispatcher,
        parse_err + 'dataset \'' + dataset_name + '\' missing key \'' + KEY_MATRIX + '\' (required)'
    ): return None, dataset_name

    dataset_matrix = data_subset_config[KEY_MATRIX]

    # assert matrix name is available in the dataset
    if not assert_is_one_of_list(
        dataset_matrix,
        dataset.get_available_matrices(),
        event_dispatcher,
        parse_err + 'unknown matrix \'' + str(dataset_matrix) + '\''
    ): return None, dataset_name + ' ' + str(dataset_matrix)

    # parse dataset filter passes

    dataset_filter_passes = parse_data_filter_passes(
        dataset_name + ' ' + dataset_matrix,
        data_subset_config,
        (dataset, dataset_matrix),
        data_filter_factory,
        event_dispatcher
    )

    parsed_config = DataSubsetConfig(
        dataset_name,
        dataset_matrix,
        dataset_filter_passes
    )

    return parsed_config, dataset_name + ' ' + dataset_matrix


def parse_data_filter_passes(
        data_parent_name: str,
        data_parent_config: Dict[str, Any],
        dataset_pair: Tuple[Dataset, str],
        filter_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> List[FilterPassConfig]:
    """Parse a list of filter pass configurations.

    Args:
        data_parent_name: the parent name related to the filter passes that are being parsed.
        data_parent_config: the parent configuration to parse the filter passes from.
        dataset_pair: a pair consisting of the dataset and the matrix name.
        filter_factory: the filter factory containing available filters for the dataset.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        a list of parsed subgroup configurations.
    """
    # filter passes are not mandatory
    if KEY_DATA_SUBSET not in data_parent_config:
        return []

    filter_passes_config = data_parent_config[KEY_DATA_SUBSET]

    # assert filter_passes_config is a list
    if not assert_is_type(
        filter_passes_config,
        list,
        event_dispatcher,
        'PARSE WARNING: ' + data_parent_name + ' invalid \'' + KEY_DATA_SUBSET + '\' value'
    ): return []

    filter_passes = []
    # attempt to parse each filter pass
    for filter_pass_config in filter_passes_config:
        parsed_filter_pass = parse_data_filter_pass_config(
            data_parent_name,
            filter_pass_config,
            dataset_pair,
            filter_factory,
            event_dispatcher
        )
        # skip on failure
        if parsed_filter_pass is None:
            continue

        filter_passes.append(parsed_filter_pass)

    return filter_passes


def parse_data_filter_pass_config(
        parent_filter_name: str,
        filter_pass_config: Any,
        dataset_pair: Tuple[Dataset, str],
        filter_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Optional[FilterPassConfig]:
    """Parse data filter pass configuration to multiple filters.

    Args:
        parent_filter_name: the filter parent name related to the filter pass that is being parsed.
        filter_pass_config: the filter pass configuration to parse.
        dataset_pair: a pair consisting of the dataset and the matrix name.
        filter_factory: the filter factory containing available filters for the dataset.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration or None on failure.
    """
    # assert filter_pass_config is a dict
    if not assert_is_type(
        filter_pass_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: ' + parent_filter_name + ' invalid ' + KEY_DATA_FILTER_PASS + ' value'
    ): return None

    # assert KEY_DATA_FILTER_PASS is present
    if not assert_is_key_in_dict(
        KEY_DATA_FILTER_PASS,
        filter_pass_config,
        event_dispatcher,
        'PARSE WARNING: ' + parent_filter_name + ' missing key \'' + KEY_DATA_FILTER_PASS + '\''
    ): return None

    filter_config = filter_pass_config[KEY_DATA_FILTER_PASS]

    dataset_filter_factory = filter_factory.get_factory(dataset_pair[0].get_name())
    matrix_filter_factory = dataset_filter_factory.get_factory(dataset_pair[1])

    # parse filter configurations as objects
    parsed_config_objs = parse_config_object_list(
        parent_filter_name,
        KEY_DATA_FILTER_PASS,
        filter_config,
        matrix_filter_factory,
        event_dispatcher
    )

    # convert object to filter configurations
    filter_config_list = []
    for (filter_config, _) in parsed_config_objs:
        filter_config_list.append(FilterConfig(
            filter_config.name,
            filter_config.params,
        ))

    # assert filter pass has entries available
    if not assert_is_container_not_empty(
        filter_config_list,
        event_dispatcher,
        'PARSE WARNING: ' + parent_filter_name + ' has no filters, skipping...'
    ): return None

    return FilterPassConfig(filter_config_list)
