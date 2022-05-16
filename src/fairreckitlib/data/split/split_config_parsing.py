"""This module contains a parser for the dataset splitting configuration.

Functions:

    parse_data_split_config: parse split configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

from ...core.config_constants import KEY_NAME, KEY_PARAMS
from ...core.config_params import ConfigOptionParam, ConfigValueParam
from ...core.event_dispatcher import EventDispatcher
from ...core.factories import Factory
from ...core.parsing.parse_assert import assert_is_type, assert_is_key_in_dict
from ...core.parsing.parse_event import ON_PARSE
from ...core.parsing.parse_params import parse_config_param, parse_config_parameters
from ..set.dataset import Dataset
from .split_config import SplitConfig
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
    parsed_config = SplitConfig(
        DEFAULT_SPLIT_NAME,
        split_factory.create_params(DEFAULT_SPLIT_NAME).get_defaults(),
        DEFAULT_SPLIT_TEST_RATIO
    )

    # dataset splitting is optional
    if KEY_SPLITTING not in dataset_config:
        event_dispatcher.dispatch(
            ON_PARSE,
            msg='PARSE WARNING: dataset ' + dataset.name + ' missing key \'' +
                KEY_SPLITTING + '\'',
            default=parsed_config
        )
        return parsed_config

    split_config = dataset_config[KEY_SPLITTING]

    # assert split_config is a dict
    if not assert_is_type(
        split_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset.name + ' invalid splitting value',
        default=parsed_config
    ): return parsed_config

    # parse splitting test ratio
    success, test_ratio = parse_config_param(
        split_config,
        dataset.name + ' ' + KEY_SPLITTING,
        ConfigValueParam(
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
        dataset.name + ' ' + KEY_SPLITTING,
        ConfigOptionParam(
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
        'PARSE WARNING: dataset ' + dataset.name + ' ' + KEY_SPLITTING + ' missing key \'' +
        KEY_PARAMS + '\'',
        default=parsed_config.params
    ):
        # parse the splitter parameters
        parsed_config.params = parse_config_parameters(
            split_config[KEY_PARAMS],
            dataset.name,
            split_params,
            event_dispatcher
        )

    return parsed_config
