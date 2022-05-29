"""This module contains functionality to parse configuration parameter(s).

Functions:

    parse_config_parameters: parse multiple parameters.
    parse_config_param: parse a single parameter.
    trim_config_params: trim unnecessary params that are not present in the config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

from ..events.event_dispatcher import EventDispatcher
from ..params.config_base_param import ConfigParam
from ..params.config_parameters import ConfigParameters
from .parse_assert import assert_is_container_not_empty, assert_is_type
from .parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from .parse_event import ON_PARSE, ParseEventArgs


def parse_config_parameters(
        params_config: Dict[str, Any],
        parent_name: str,
        parameters: ConfigParameters,
        event_dispatcher: EventDispatcher) -> Dict[str, Any]:
    """Parse the object's parameters configuration.

    Args:
        params_config: dictionary with the configuration to parse.
        parent_name: name of the parent related to the parameters' configuration.
        parameters: the configuration parameters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed params configuration as key-value pairs.
    """
    # start with parameter defaults
    parsed_params = parameters.get_defaults()

    # assert params_config is a dict
    if not assert_is_type(
        params_config,
        dict,
        event_dispatcher,
        'PARSE WARNING: ' + parent_name + ' invalid params value',
        default_value=parsed_params
    ): return parsed_params

    # remove unnecessary parameters from configuration
    params_config = trim_config_params(
        params_config,
        parent_name,
        parameters,
        event_dispatcher
    )

    # assert params_config has entries left after trimming
    if not assert_is_container_not_empty(
        params_config,
        event_dispatcher,
        'PARSE WARNING: ' + parent_name + ' params is empty',
        default_value=parsed_params
    ): return parsed_params

    # parse params_config entries
    for param_name, _ in parsed_params.items():
        success, value = parse_config_param(
            params_config,
            parent_name,
            parameters.get_param(param_name),
            event_dispatcher
        )

        # replace defaults on success
        if success:
            parsed_params[param_name] = value

    return parsed_params


def parse_config_param(
        params_config: Dict[str, Any],
        parent_name: str,
        param: ConfigParam,
        event_dispatcher: EventDispatcher) -> Tuple[bool, Any]:
    """Parse a parameter from the specified configuration.

    Args:
        params_config: dictionary with the parameters' configuration.
        parent_name: name of the parent related to the parameters' configuration.
        param: the parameter that is being parsed.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        whether the parsing succeeded and the parsed (validated) value.
    """
    param_default = param.default_value

    # assert param_name is present in the configuration
    if not assert_is_key_in_dict(
        param.name,
        params_config,
        event_dispatcher,
        'PARSE WARNING: ' + parent_name + ' missing param for \'' + param.name + '\'',
        default_value=param_default
    ): return False, param_default

    config_value = params_config[param.name]
    # validate the configuration value
    success, value, error_msg = param.validate_value(config_value)

    if not success:
        event_dispatcher.dispatch(ParseEventArgs(
            ON_PARSE,
            'PARSE WARNING: ' + parent_name + ' invalid param \'' + param.name + '\'' +
            '\n\t' + error_msg,
            actual_type=config_value,
            default_value=value
        ))
    # validation succeeded but extra info is available
    elif len(error_msg) > 0:
        event_dispatcher.dispatch(ParseEventArgs(
            ON_PARSE,
            'PARSE WARNING: ' + parent_name + ' modified param \'' + param.name + '\'' +
            '\n\t' + error_msg,
            actual_type=config_value
        ))

    return success, value


def trim_config_params(
        params_config: Dict[str, Any],
        parent_name: str,
        parameters: ConfigParameters,
        event_dispatcher: EventDispatcher) -> Dict[str, Any]:
    """Trim the parameters from the specified configuration.

    Removes unnecessary parameters that are not present in the
    original config parameter list.

    Args:
        params_config: dictionary with the parameters' configuration.
        parent_name: name of the parent related to the parameters' configuration.
        parameters: the configuration parameters.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the dictionary with the trimmed parameters.
    """
    trimmed_config = {}

    for param_name, param_value in params_config.items():
        if assert_is_one_of_list(
            param_name,
            parameters.get_param_names(),
            event_dispatcher,
            'PARSE WARNING: ' + parent_name + ' unknown parameter \'' + param_name + '\''
        ): trimmed_config[param_name] = param_value

    return trimmed_config
