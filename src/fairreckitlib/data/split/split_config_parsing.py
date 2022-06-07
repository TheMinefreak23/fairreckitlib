"""This module contains a parser for the dataset splitting configuration.

Functions:

    parse_data_split_config: parse split configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

from ...core.config.config_factories import Factory
from ...core.config.config_option_param import ConfigSingleOptionParam
from ...core.config.config_value_param import ConfigNumberParam
from ...core.core_constants import KEY_NAME, KEY_PARAMS
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_assert import assert_is_type, assert_is_key_in_dict
from ...core.parsing.parse_config_params import parse_config_param, parse_config_parameters
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs
from ..set.dataset import Dataset
from .split_config import SplitConfig, create_default_split_config
from .split_constants import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from .split_constants import DEFAULT_SPLIT_TEST_RATIO, DEFAULT_SPLIT_NAME


def parse_data_split_config(
        dataset_config: Dict[str, Any],
        dataset: Dataset,
        split_factory: Factory,
        event_dispatcher: EventDispatcher) -> SplitConfig:
    """Parse a dataset splitting configuration.

    Args:
        dataset_config: the dataset's total configuration.
        dataset: the dataset related to the splitting configuration.
        split_factory: the split factory containing available splitters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration or None on failure.
    """
    parsed_config = create_default_split_config()

    # dataset splitting is optional
    if KEY_SPLITTING not in dataset_config:
        event_dispatcher.dispatch(ParseEventArgs(
            ON_PARSE,
            'PARSE WARNING: dataset ' + dataset.get_name() + ' missing key \'' +
            KEY_SPLITTING + '\'',
            default_value=parsed_config
        ))
        return parsed_config

    split_config = dataset_config[KEY_SPLITTING]

    # assert split_config is a dict
    if not assert_is_type(
        split_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset.get_name() + ' invalid splitting value',
        default_value=parsed_config
    ): return parsed_config

    # parse splitting test ratio
    success, test_ratio = parse_config_param(
        split_config,
        dataset.get_name() + ' ' + KEY_SPLITTING,
        ConfigNumberParam(
            KEY_SPLIT_TEST_RATIO,
            float,
            DEFAULT_SPLIT_TEST_RATIO,
            (0.01, 0.99)
        ),
        event_dispatcher
    )
    if success:
        parsed_config.test_ratio = test_ratio

    # parse splitting name
    success, split_name = parse_config_param(
        split_config,
        dataset.get_name() + ' ' + KEY_SPLITTING,
        ConfigSingleOptionParam(
            KEY_NAME,
            str,
            DEFAULT_SPLIT_NAME,
            split_factory.get_available_names()
        ),
        event_dispatcher
    )
    if success:
        parsed_config.name = split_name

    split_params = split_factory.create_params(split_name)
    parsed_config.params = split_params.get_defaults()

    # assert KEY_PARAMS is present
    # skip when the splitter has no parameters at all
    if split_params.get_num_params() > 0 and assert_is_key_in_dict(
        KEY_PARAMS,
        split_config,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset.get_name() + ' ' + KEY_SPLITTING + ' missing key \'' +
        KEY_PARAMS + '\'',
        default_value=parsed_config.params
    ):
        # parse the splitter parameters
        parsed_config.params = parse_config_parameters(
            split_config[KEY_PARAMS],
            dataset.get_name(),
            split_params,
            event_dispatcher
        )

    return parsed_config
