"""This module contains a parser for the dataset rating conversion configuration.

Functions:

    parse_data_convert_config: parse convert configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Optional

from ...core.config.config_factories import Factory
from ...core.config.config_option_param import ConfigSingleOptionParam
from ...core.core_constants import KEY_NAME, KEY_PARAMS
from ...core.events.event_dispatcher import EventDispatcher
from ...core.parsing.parse_assert import assert_is_type, assert_is_key_in_dict
from ...core.parsing.parse_config_params import parse_config_param, parse_config_parameters
from ...core.parsing.parse_event import ON_PARSE, ParseEventArgs
from ..set.dataset import Dataset
from .convert_config import ConvertConfig
from .convert_constants import KEY_RATING_CONVERTER


def parse_data_convert_config(
        dataset_config: Dict[str, Any],
        dataset: Dataset,
        matrix_name: str,
        converter_factory: Factory,
        event_dispatcher: EventDispatcher) -> Optional[ConvertConfig]:
    """Parse a dataset rating converter configuration.

    Args:
        dataset_config: the dataset's total configuration.
        dataset: the dataset related to the converter configuration.
        matrix_name: the dataset's matrix name to use.
        converter_factory: the converter factory containing available converters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration or None on failure.
    """
    parsed_config = None

    # dataset rating conversion is optional
    if KEY_RATING_CONVERTER not in dataset_config:
        event_dispatcher.dispatch(ParseEventArgs(
            ON_PARSE,
            'PARSE WARNING: dataset ' + dataset.get_name() + ' missing key \'' +
            KEY_RATING_CONVERTER + '\'',
            default_value=parsed_config
        ))
        return parsed_config

    convert_config = dataset_config[KEY_RATING_CONVERTER]

    # assert convert_config is a dict
    if not assert_is_type(
        convert_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset.get_name() + ' invalid rating conversion value',
        default_value=parsed_config
    ): return parsed_config

    # parse converter name
    success, converter_name = parse_config_param(
        convert_config,
        dataset.get_name() + ' ' + KEY_RATING_CONVERTER,
        ConfigSingleOptionParam(
            KEY_NAME,
            str,
            'None',
            converter_factory.get_available_names()
        ),
        event_dispatcher
    )
    if not success:
        return parsed_config

    convert_params = converter_factory.create_params(converter_name)
    upper_bound = convert_params.get_param('upper_bound')
    if upper_bound is not None:
        # update upper_bound param's default value to match the dataset's max rating
        upper_bound.default_value = dataset.get_matrix_config(matrix_name).rating_max

    parsed_config = ConvertConfig(
        converter_name,
        convert_params.get_defaults()
    )

    # assert KEY_PARAMS is present
    # skip when the converter has no parameters at all
    if convert_params.get_num_params() > 0 and assert_is_key_in_dict(
        KEY_PARAMS,
        convert_config,
        event_dispatcher,
        'PARSE WARNING: dataset ' + dataset.get_name() + ' ' + KEY_RATING_CONVERTER +
        ' missing key \'' + KEY_PARAMS + '\'',
        default_value=parsed_config.params
    ):
        # parse the converter parameters
        parsed_config.params = parse_config_parameters(
            convert_config[KEY_PARAMS],
            dataset.get_name(),
            convert_params,
            event_dispatcher
        )

    return parsed_config
