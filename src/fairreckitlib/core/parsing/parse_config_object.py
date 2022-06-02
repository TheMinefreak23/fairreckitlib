"""This module contains functionality to parse configuration object name and parameters.

Functions:

    parse_config_object: parse an object name and parameters configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple, Union

from ..config.config_factories import Factory
from ..config.config_object import ObjectConfig
from ..core_constants import KEY_NAME, KEY_PARAMS
from ..events.event_dispatcher import EventDispatcher
from .parse_assert import assert_is_key_in_dict, assert_is_one_of_list, assert_is_type
from .parse_config_params import parse_config_parameters


def parse_config_object(
        obj_type_name: str,
        obj_config: Dict[str, Any],
        obj_factory: Factory,
        event_dispatcher: EventDispatcher) -> Union[Tuple[ObjectConfig, str], Tuple[None, None]]:
    """Parse an object name and parameters configuration.

    Args:
        obj_type_name: name of the object type.
        obj_config: dictionary with the object's configuration.
        obj_factory: the object factory related to the object config.
        event_dispatcher: to dispatch the parse event on failure.

    Returns:
        the parsed configuration and object name or None on failure.
    """
    # assert obj_config is a dict
    if not assert_is_type(
        obj_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: ' + obj_factory.get_name() + ' ' + obj_type_name + ' invalid config value'
    ): return None, None

    # assert obj name is present
    if not assert_is_key_in_dict(
        KEY_NAME,
        obj_config,
        event_dispatcher,
        'PARSE ERROR: ' + obj_factory.get_name() + ' ' + obj_type_name +
        ' missing key \'' + KEY_NAME + '\''
    ): return None, None

    obj_name = obj_config[KEY_NAME]

    # assert object name is available in the object factory
    if not assert_is_one_of_list(
        obj_name,
        obj_factory.get_available_names(),
        event_dispatcher,
        'PARSE ERROR: ' + obj_factory.get_name() + ' ' + obj_type_name +
        ' unknown name: \'' + str(obj_name) + '\''
    ): return None, obj_name

    obj_config_params = obj_factory.create_params(obj_name)
    obj_params = obj_config_params.get_defaults()

    # assert KEY_PARAMS is present
    # skip when the object has no parameters at all
    if obj_config_params.get_num_params() > 0 and assert_is_key_in_dict(
        KEY_PARAMS,
        obj_config,
        event_dispatcher,
        'PARSE WARNING: ' + obj_type_name + ' ' + obj_name + ' missing key \'' + KEY_PARAMS + '\'',
        default_value=obj_params
    ):
        # parse the object parameters
        obj_params = parse_config_parameters(
            obj_config[KEY_PARAMS],
            obj_name,
            obj_config_params,
            event_dispatcher
        )

    parsed_config = ObjectConfig(
        obj_name,
        obj_params
    )

    return parsed_config, obj_name
