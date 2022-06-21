"""This module contains a parser for the dataset splitting configuration.

Functions:

    parse_data_split_config: parse split configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

from ...core.config.config_factories import Factory
from ...core.config.config_value_param import ConfigNumberParam
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_config_object import parse_config_object
from ...core.parsing.parse_config_params import parse_config_param
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs
from ..set.dataset import Dataset
from .split_config import SplitConfig, create_default_split_config
from .split_constants import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from .split_constants import DEFAULT_SPLIT_TEST_RATIO, MIN_TEST_RATIO, MAX_TEST_RATIO


def parse_data_split_config(
        dataset_config: Dict[str, Any],
        dataset: Dataset,
        matrix_name: str,
        split_factory: Factory,
        event_dispatcher: EventDispatcher) -> SplitConfig:
    """Parse a dataset splitting configuration.

    Args:
        dataset_config: the dataset's total configuration.
        dataset: the dataset related to the splitting configuration.
        matrix_name: the dataset's matrix name that is used.
        split_factory: the split factory containing available splitters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration or None on failure.
    """
    default_config = create_default_split_config()

    # dataset splitting is optional
    if KEY_SPLITTING not in dataset_config:
        event_dispatcher.dispatch(ParseEventArgs(
            ON_PARSE,
            'PARSE WARNING: dataset ' + dataset.get_name() + ' \'' + matrix_name +
            '\' missing key \'' + KEY_SPLITTING + '\'',
            default_value=default_config
        ))
        return default_config

    split_config = dataset_config[KEY_SPLITTING]

    splitter, _ = parse_config_object(
        'dataset ' + dataset.get_name() + ' \'' + matrix_name + '\' splitter',
        split_config,
        split_factory,
        event_dispatcher,
        default_config=default_config
    )
    if not bool(splitter):
        return default_config

    # parse splitting test ratio
    _, test_ratio = parse_config_param(
        split_config,
        'dataset ' + dataset.get_name() + ' \'' + matrix_name +
        '\' splitter \'' + splitter.name + '\'',
        ConfigNumberParam(
            KEY_SPLIT_TEST_RATIO,
            float,
            DEFAULT_SPLIT_TEST_RATIO,
            (MIN_TEST_RATIO, MAX_TEST_RATIO)
        ),
        event_dispatcher
    )

    return SplitConfig(splitter.name, splitter.params, test_ratio)
