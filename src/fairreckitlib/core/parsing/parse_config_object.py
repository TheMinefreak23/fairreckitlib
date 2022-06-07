"""This module contains functionality to parse configuration object name and parameters.

Functions:

    parse_config_object: parse an object name and parameters configuration.
    parse_config_object_list: parse a list of config objects

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Tuple, Union

from ..config.config_factories import Factory, GroupFactory, resolve_factory
from ..config.config_object import ObjectConfig
from ..config.config_base_param import ConfigParam
from ..core_constants import KEY_NAME, KEY_PARAMS
from ..events.event_dispatcher import EventDispatcher
from .parse_assert import \
    assert_is_container_not_empty, assert_is_key_in_dict, assert_is_one_of_list, assert_is_type
from .parse_config_params import parse_config_parameters
from .parse_event import ON_PARSE, ParseEventArgs


def parse_config_object(
        obj_type_name: str,
        obj_config: Dict[str, Any],
        obj_factory: Union[Factory, GroupFactory],
        event_dispatcher: EventDispatcher,
        *,
        params: Dict[str, ConfigParam]=None) -> Union[Tuple[ObjectConfig, str], Tuple[None, None]]:
    """Parse an object name and parameters configuration.

    Args:
        obj_type_name: name of the object type.
        obj_config: dictionary with the object's configuration.
        obj_factory: the object (group) factory related to the object config.
        event_dispatcher: to dispatch the parse event on failure.
        params: dictionary with params that will override validation of existing config params.

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
    # attempt to resolve the factory that creates the object
    obj_create_factory = resolve_factory(obj_name, obj_factory)

    # assert object name is available in the object factory
    if not assert_is_one_of_list(
        obj_name,
        [] if obj_create_factory is None else obj_create_factory.get_available_names(),
        event_dispatcher,
        'PARSE ERROR: ' + obj_factory.get_name() + ' ' + obj_type_name +
        ' unknown name: \'' + str(obj_name) + '\''
    ): return None, obj_name

    obj_config_params = obj_create_factory.create_params(obj_name)
    # override any present config params
    if params is not None:
        for param_name, config_param in params.items():
            if param_name in obj_config_params.params:
                obj_config_params.params[param_name] = config_param

    # start with object default params
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


def parse_config_object_list(
        obj_category_name: str,
        obj_type_name: str,
        obj_config_list: List[Dict[str, Any]],
        obj_factory: GroupFactory,
        event_dispatcher: EventDispatcher,
        *,
        params: Dict[str, ConfigParam]=None) -> List[Tuple[ObjectConfig, Dict[str, Any]]]:
    """Parse the object configurations for the specified category name.

    Args:
        obj_category_name: name of the object category.
        obj_type_name: name of the object type.
        obj_config_list: list of dictionaries with the object's configuration.
        obj_factory: the object (group) factory related to the object config.
        event_dispatcher: to dispatch the parse event on failure.
        params: dictionary with params that will override validation of existing config params.

    Returns:
        a tuple list of the successfully parsed ObjectConfig's and their corresponding dictionary.
    """
    parsed_objs = []

    # assert obj_config_list is a list
    if not assert_is_type(
        obj_config_list,
        list,
        event_dispatcher,
        'PARSE ERROR: ' + obj_category_name + ' invalid config value',
        default_value=parsed_objs
    ): return parsed_objs

    # assert obj_config_list has entries
    if not assert_is_container_not_empty(
        obj_config_list,
        event_dispatcher,
        'PARSE WARNING: ' + obj_category_name + ' ' + obj_type_name + ' list is empty'
    ): return parsed_objs

    # parse obj config list entries
    for obj_config in obj_config_list:
        obj, obj_name = parse_config_object(
            obj_type_name,
            obj_config,
            obj_factory,
            event_dispatcher,
            params=params
        )
        # skip on failure
        if obj is None:
            event_dispatcher.dispatch(ParseEventArgs(
                ON_PARSE,
                'PARSE WARNING: failed to parse ' + obj_category_name + ' ' + obj_type_name +
                ' \'' + str(obj_name) + '\', skipping...'
            ))
            continue

        parsed_objs.append((ObjectConfig(obj.name, obj.params), obj_config))

    return parsed_objs
