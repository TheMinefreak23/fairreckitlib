"""This module contains a parser for the dataset configuration.

Functions:

    parse_data_config: parse dataset matrices from the experiment configuration.
    parse_data_matrix_config: parse dataset matrix configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_assert import \
    assert_is_type, assert_is_container_not_empty, assert_is_key_in_dict
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs
from ..data_factory import KEY_DATA
from ..filter.filter_config_parsing import parse_data_subset_config
from ..filter.filter_constants import KEY_DATA_SUBSET
from ..ratings.convert_constants import KEY_RATING_CONVERTER
from ..ratings.convert_config_parsing import parse_data_convert_config
from ..set.dataset_registry import DataRegistry
from ..split.split_constants import KEY_SPLITTING
from ..split.split_config_parsing import parse_data_split_config
from .data_config import DataMatrixConfig


def parse_data_config(
        experiment_config: Dict[str, Any],
        data_registry: DataRegistry,
        data_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Optional[List[DataMatrixConfig]]:
    """Parse all dataset configurations.

    Args:
        experiment_config: the experiment's total configuration.
        data_registry: the data registry containing the available datasets.
        data_factory: factory with available data modifier factories.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        a list of parsed DatasetConfig's or None when empty.
    """
    # assert KEY_DATA is present
    if not assert_is_key_in_dict(
        KEY_DATA,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + KEY_DATA + '\' (required)'
    ): return None

    data_matrices_config = experiment_config[KEY_DATA]

    # assert data_matrices_config is a list
    if not assert_is_type(
        data_matrices_config,
        list,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + KEY_DATA + '\''
    ): return None

    # assert data_matrices_config has list entries
    if not assert_is_container_not_empty(
        data_matrices_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + KEY_DATA + '\' is empty'
    ): return None

    parsed_matrices = []

    # parse datasets_config list entries
    for data_matrix_config in data_matrices_config:
        data_matrix, data_matrix_name = parse_data_matrix_config(
            data_matrix_config,
            data_registry,
            data_factory,
            event_dispatcher
        )
        # skip on failure
        if data_matrix is None:
            event_dispatcher.dispatch(ParseEventArgs(
                ON_PARSE,
                'PARSE WARNING: failed to parse data matrix \'' +
                str(data_matrix_name) + '\', skipping...'
            ))
            continue

        parsed_matrices.append(data_matrix)

    # final check to verify at least one data matrix got parsed
    if not assert_is_container_not_empty(
        parsed_matrices,
        event_dispatcher,
        'PARSE ERROR: missing experiment data matrices'
    ): return None

    return parsed_matrices


def parse_data_matrix_config(
        data_matrix_config: Any,
        data_registry: DataRegistry,
        data_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Union[Tuple[DataMatrixConfig, str],Tuple[None, None]]:
    """Parse a data matrix configuration.

    Args:
        data_matrix_config: the data matrix configuration.
        data_registry: the data registry containing the available datasets.
        data_factory: factory with available data modifier factories.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        parsed_config: the parsed configuration or None on failure.
        dataset_name: the name of the parsed dataset or None on failure.
    """
    # assert data_matrix_config is a dict
    if not assert_is_type(
        data_matrix_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid dataset matrix entry'
    ): return None, None

    dataset_subset, dataset_matrix_name = parse_data_subset_config(
        data_matrix_config,
        data_registry,
        data_factory.get_factory(KEY_DATA_SUBSET),
        event_dispatcher
    )
    if not dataset_subset:
        return None, dataset_matrix_name

    dataset = data_registry.get_set(dataset_subset.dataset)

    # parse dataset matrix rating converter
    dataset_rating_modifier = parse_data_convert_config(
        data_matrix_config,
        dataset,
        dataset_subset.matrix,
        data_factory.get_factory(KEY_RATING_CONVERTER),
        event_dispatcher
    )

    # parse dataset matrix splitter
    dataset_splitting = parse_data_split_config(
        data_matrix_config,
        dataset,
        dataset_subset.matrix,
        data_factory.get_factory(KEY_SPLITTING),
        event_dispatcher
    )

    parsed_config = DataMatrixConfig(
        dataset_subset.dataset,
        dataset_subset.matrix,
        dataset_subset.filter_passes,
        dataset_rating_modifier,
        dataset_splitting
    )

    return parsed_config, dataset_matrix_name
