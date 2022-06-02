"""This module contains a parser for the dataset configuration.

Functions:

    parse_data_config: parse (multiple) dataset configurations.
    parse_dataset_config: parse dataset configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_assert import assert_is_type, assert_is_container_not_empty
from ...core.parsing.parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs
from ..data_factory import KEY_DATA
from ..ratings.convert_constants import KEY_RATING_CONVERTER
from ..ratings.convert_config_parsing import parse_data_convert_config
from ..set.dataset_constants import KEY_DATASET, KEY_MATRIX
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
    # assert KEY_DATASETS is present
    if not assert_is_key_in_dict(
        KEY_DATA,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + KEY_DATA + '\' (required)'
    ): return None

    datasets_config = experiment_config[KEY_DATA]

    # assert datasets_config is a list
    if not assert_is_type(
        datasets_config,
        list,
        event_dispatcher,
        'PARSE ERROR: invalid experiment value for key \'' + KEY_DATA + '\''
    ): return None

    # assert datasets_config has list entries
    if not assert_is_container_not_empty(
        datasets_config,
        event_dispatcher,
        'PARSE ERROR: experiment \'' + KEY_DATA + '\' is empty'
    ): return None

    parsed_config = []

    # parse datasets_config list entries
    for _, dataset_config in enumerate(datasets_config):
        dataset, dataset_name = parse_dataset_config(
            dataset_config,
            data_registry,
            data_factory,
            event_dispatcher
        )
        # skip on failure
        if dataset is None:
            event_dispatcher.dispatch(ParseEventArgs(
                ON_PARSE,
                'PARSE WARNING: failed to parse dataset \'' +
                str(dataset_name) + '\', skipping...'
            ))
            continue

        parsed_config.append(dataset)

    if not assert_is_container_not_empty(
        parsed_config,
        event_dispatcher,
        'PARSE ERROR: no experiment ' + KEY_DATA + ' specified'
    ): return None

    return parsed_config


def parse_dataset_config(
        dataset_config: Dict[str, Any],
        data_registry: DataRegistry,
        data_factory: GroupFactory,
        event_dispatcher: EventDispatcher) -> Union[Tuple[DataMatrixConfig, str],Tuple[None, None]]:
    """Parse a dataset configuration.

    Args:
        dataset_config: the dataset's configuration.
        data_registry: the data registry containing the available datasets.
        data_factory: factory with available data modifier factories.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        parsed_config: the parsed configuration or None on failure.
        dataset_name: the name of the parsed dataset or None on failure.
    """
    # assert dataset_config is a dict
    if not assert_is_type(
        dataset_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid dataset entry'
    ): return None, None

    # assert dataset name is present
    if not assert_is_key_in_dict(
        KEY_DATASET,
        dataset_config,
        event_dispatcher,
        'PARSE ERROR: missing key \'' + KEY_DATASET + '\' (required)'
    ): return None, None

    dataset_name = dataset_config[KEY_DATASET]

    # assert dataset name is available in the data registry
    if not assert_is_one_of_list(
        dataset_name,
        data_registry.get_available_sets(),
        event_dispatcher,
        'PARSE ERROR: unknown dataset name \'' + str(dataset_name) + '\''
    ): return None, dataset_name

    dataset = data_registry.get_set(dataset_name)

    # assert matrix name is present
    if not assert_is_key_in_dict(
        KEY_MATRIX,
        dataset_config,
        event_dispatcher,
        'PARSE ERROR: missing key \'' + KEY_MATRIX + '\' (required)'
    ): return None, dataset_name

    dataset_matrix = dataset_config[KEY_MATRIX]

    # assert matrix name is available in the dataset
    if not assert_is_one_of_list(
        dataset_matrix,
        dataset.get_available_matrices(),
        event_dispatcher,
        'PARSE ERROR: unknown dataset matrix \'' + str(dataset_matrix) + '\''
    ): return None, dataset_name

    # TODO parse this
    dataset_prefilters = []

    # parse dataset rating converter
    dataset_rating_modifier = parse_data_convert_config(
        dataset_config,
        dataset,
        dataset_matrix,
        data_factory.get_factory(KEY_RATING_CONVERTER),
        event_dispatcher
    )

    # parse dataset split
    dataset_splitting = parse_data_split_config(
        dataset_config,
        dataset,
        data_factory.get_factory(KEY_SPLITTING),
        event_dispatcher
    )

    parsed_config = DataMatrixConfig(
        dataset_name,
        dataset_matrix,
        dataset_prefilters,
        dataset_rating_modifier,
        dataset_splitting
    )

    return parsed_config, dataset_name
