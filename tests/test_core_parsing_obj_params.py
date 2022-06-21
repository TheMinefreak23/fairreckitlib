"""This module tests the object (parameters) parsing functionality.

Functions:

    create_dummy_config_params: factory compatible dummy object configuration parameters.
    create_dummy_obj: factory compatible dummy object creation function.
    test_trim_config_params: test trimming of (unused) configuration parameters.
    test_parse_config_param: test parsing of a single configuration parameter.
    test_parse_config_parameters: test parsing of the configuration parameters as a whole.
    test_parse_config_obj: test parsing of an object name and parameters configuration.
    test_parse_config_obj_list: test parsing of a list of configuration objects.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import pytest

from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.config.config_factories import Factory, GroupFactory
from src.fairreckitlib.core.config.config_option_param import create_bool_param
from src.fairreckitlib.core.config.config_parameters import ConfigParameters
from src.fairreckitlib.core.parsing.parse_config_object import \
    parse_config_object, parse_config_object_list
from src.fairreckitlib.core.parsing.parse_config_params import \
    parse_config_parameters, parse_config_param, trim_config_params

BOOL_PARAM_NAME = 'bool'
INT_PARAM_NAME = 'int'
FLOAT_PARAM_NAME = 'float'
SEED_PARAM_NAME = 'seed'
UNKNOWN_PARAM_NAME = 'unknown'

BOOL_PARAM_VALUE = True

INVALID_CONTAINER_TYPES = \
    [None, True, False, 0, 0.0, 'a', [], {UNKNOWN_PARAM_NAME: 'dict'}, {'set'}]


def create_dummy_config_params() -> ConfigParameters:
    """Create dummy configuration parameters to use for testing."""
    config_params = ConfigParameters()
    config_params.add_random_seed(SEED_PARAM_NAME)
    config_params.add_bool(BOOL_PARAM_NAME, BOOL_PARAM_VALUE)
    config_params.add_number(INT_PARAM_NAME, int, 10, (0, 100))
    config_params.add_number(FLOAT_PARAM_NAME, float, 10.0, (0.0, 100.0))
    config_params.add_range('int_range', int, (1, 2), (0, 3))
    config_params.add_range('float_range', float, (1.0, 2.0), (0.0, 3.0))
    config_params.add_single_option('single', str, 'a', ['a', 'b', 'c'])
    config_params.add_multi_option('multi', ['a', 'c'], ['a', 'b', 'c'])
    return config_params


def create_dummy_obj(name: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Create dummy object function not used, only to initialize the object factory."""
    return name, params


def test_trim_config_params(parse_event_dispatcher: EventDispatcher) -> None:
    """Test config parameter trimming with and without an unknown parameter."""
    config_params = create_dummy_config_params()
    param_defaults = config_params.get_defaults()

    # test trimming without an unknown parameter.
    trimmed_params = trim_config_params(param_defaults, '', config_params, parse_event_dispatcher)

    assert len(trimmed_params) == len(param_defaults), 'did not expect parameters to be trimmed.'
    for param_name in config_params.get_param_names():
        assert param_name in trimmed_params, 'expected original parameter to be present.'

    # test trimming with an unknown parameter.
    param_defaults[UNKNOWN_PARAM_NAME] = False
    trimmed_params = trim_config_params(param_defaults, '', config_params, parse_event_dispatcher)

    assert len(trimmed_params) + 1 == len(param_defaults), 'expected parameters to be trimmed.'
    for param_name in config_params.get_param_names():
        assert param_name in trimmed_params, 'expected original parameter to be present.'
    with pytest.raises(KeyError):
        # test failure to retrieve unknown param after trimming
        _ = trimmed_params[UNKNOWN_PARAM_NAME]


def test_parse_config_param(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing of a single configuration parameter."""
    config_params = create_dummy_config_params()

    for param_name in config_params.get_param_names():
        params = config_params.get_defaults()
        config_param = config_params.get_param(param_name)

        # test success for happy path (using the default value)
        success, parsed_val = parse_config_param(params, '', config_param, parse_event_dispatcher)
        assert success and parsed_val == config_param.default_value, \
            'expected parameter default to succeed when parsing.'

        # test success for int/float casting,
        # for simplicity only primitives as casting is already tested in config param validation
        cast_switch = {INT_PARAM_NAME: float, FLOAT_PARAM_NAME: int}
        if param_name in cast_switch:
            params[param_name] = cast_switch[param_name](params[param_name])
            assert parse_config_param(params, 'Casting', config_param, parse_event_dispatcher)[0],\
                'expected casting between int/floats to succeed when parsing.'

        # test failure for invalid param value,
        # None is not accepted for any config param class, except for a random seed param
        params[param_name] = -1 if param_name == SEED_PARAM_NAME else None
        success, parsed_val = parse_config_param(params, '', config_param, parse_event_dispatcher)
        assert not success, 'expected invalid parameter value to fail on parsing.'
        assert parsed_val == (0 if param_name == SEED_PARAM_NAME else config_param.default_value),\
            'expected invalid parameter value to fail on parsing.'

        # test failure for missing param value
        del params[param_name]
        assert not parse_config_param(
            params,
            'param \'' + param_name + '\' was removed =>',
            config_param,
            parse_event_dispatcher
        )[0], 'expected missing parameter to fail on parsing.'


def test_parse_config_parameters(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing of the configuration parameters as a whole."""
    config_params = create_dummy_config_params()

    # test failure for parsing various types, including a dict that will be empty after trimming
    for params in INVALID_CONTAINER_TYPES:
        params = parse_config_parameters(params, str(params), config_params,parse_event_dispatcher)
        assert params == config_params.get_defaults(), \
            'expected config parameters parsing to fail and return the defaults'

    # test success on updated param values,
    # using boolean for simplicity as the individual param parsing/validations are already tested
    params = config_params.get_defaults()
    params[BOOL_PARAM_NAME] = not BOOL_PARAM_VALUE
    params = parse_config_parameters(params, str(params), config_params, parse_event_dispatcher)
    assert params != config_params.get_defaults(), \
        'expected config parameters parsing to succeed and not be the same as the defaults.'
    assert params[BOOL_PARAM_NAME] == (not BOOL_PARAM_VALUE), \
        'expected the boolean config parameter value to be flipped.'


def test_parse_config_obj(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing of an object name and parameters configuration."""
    obj_type_name = 'test_obj'
    obj_name_unknown = 'unknown_obj'
    obj_name = 'obj'
    obj_factory = Factory('obj_factory')

    # test failure when the factory cannot be resolved
    parsed_config, parsed_obj_name = parse_config_object(
        obj_type_name,
        {KEY_NAME: obj_name},
        obj_factory,
        parse_event_dispatcher
    )
    assert parsed_config is None, \
        'did not expect the parsing to succeed for an unresolved factory'
    assert parsed_obj_name == obj_name, \
        'expected the name of the object to be parsed correctly'

    obj_factory.add_obj(obj_name, create_dummy_obj, create_dummy_config_params)

    # test failure on incorrect types, including an invalid dictionary
    for obj_config in INVALID_CONTAINER_TYPES:
        parsed_config, parsed_obj_name = parse_config_object(
            obj_type_name,
            obj_config,
            obj_factory,
            parse_event_dispatcher
        )
        assert parsed_config is None, 'did not expect parsed obj configuration on failure'
        assert parsed_obj_name is None, 'did not expect parsed obj name on failure'

    # test failure on dictionary with an object name that is not known in the factory
    obj_config = {KEY_NAME: obj_name_unknown}
    parsed_config, parsed_obj_name = parse_config_object(
        obj_type_name,
        obj_config,
        obj_factory,
        parse_event_dispatcher
    )
    assert parsed_config is None, 'did not expect parsed obj configuration for an unknown object'
    assert parsed_obj_name == obj_name_unknown, 'expected parsed obj name to be the same as input'

    obj_param_defaults = create_dummy_config_params().get_defaults()
    obj_config[KEY_NAME] = obj_name
    # test success on correct dictionary with and without provided params
    for i in range(0, 2):
        if i == 1:
            # add params on second iteration to see if the parsing still succeeds
            obj_config[KEY_PARAMS] = obj_param_defaults

        parsed_config, parsed_obj_name = parse_config_object(
            obj_type_name,
            obj_config,
            obj_factory,
            parse_event_dispatcher
        )
        assert parsed_obj_name == obj_name, \
            'expected parsed obj name to be the same as the known object name'
        assert parsed_config.name == obj_name, \
            'expected the parsed config obj name to be the same as the known object name'
        assert parsed_config.params == obj_param_defaults, \
            'expected the parsed config params to be the same as the defaults'

    # test object parsing to succeed with multiple layers of factory branches
    group_factory = None
    sub_group_factory = obj_factory
    for i in range(0, 10):
        if group_factory is not None:
            sub_group_factory = group_factory

        group_factory = GroupFactory('obj_group' + str(i))
        group_factory.add_factory(sub_group_factory)

        parsed_config, _ = parse_config_object(
            obj_type_name,
            obj_config,
            group_factory,
            parse_event_dispatcher
        )

        assert parsed_config is not None, \
            'expected parsing to succeed with group factory sub availability'

    # test object parsing to succeed with an override config param to replace the existing param
    override_params = {BOOL_PARAM_NAME: create_bool_param(BOOL_PARAM_NAME, not BOOL_PARAM_VALUE)}
    # remove boolean param to check return value of override config param
    del obj_config[KEY_PARAMS][BOOL_PARAM_NAME]
    parsed_config, _ = parse_config_object(
        obj_type_name,
        obj_config,
        obj_factory,
        parse_event_dispatcher,
        params=override_params
    )

    assert parsed_config is not None, \
        'expected parsing to succeed with an override param dictionary'
    assert parsed_config.params[BOOL_PARAM_NAME] == (not BOOL_PARAM_VALUE), \
        'expected override param default value instead of the original default value'


def test_parse_config_obj_list(parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing of a list of object name and parameters configurations."""
    obj_name_unknown = 'unknown_obj'
    obj_category_name = 'cat_obj'
    obj_type_name = 'test_obj'
    obj_factory = Factory('obj_factory')

    # test failure on incorrect types, including an empty list
    for obj_config_list in INVALID_CONTAINER_TYPES:
        parsed_config_list = parse_config_object_list(
            obj_category_name,
            obj_type_name,
            obj_config_list,
            obj_factory,
            parse_event_dispatcher
        )
        assert len(parsed_config_list) == 0, \
            'did not expect any object configurations to be parsed'

    obj_config_list = []
    for i in range(0, 10):
        obj_name = str(i)
        obj_factory.add_obj(obj_name, create_dummy_obj, create_dummy_config_params)
        obj_config_list.append({
            KEY_NAME: obj_name,
            KEY_PARAMS: create_dummy_config_params().get_defaults()
        })

    # test success for all object configurations
    parsed_config_list = parse_config_object_list(
        obj_category_name,
        obj_type_name,
        obj_config_list,
        obj_factory,
        parse_event_dispatcher
    )
    assert len(parsed_config_list) == len(obj_config_list), \
        'expected parsing for all objects to succeed'

    # test object list parsing to succeed with an override config param
    override_params = {BOOL_PARAM_NAME: create_bool_param(BOOL_PARAM_NAME, not BOOL_PARAM_VALUE)}
    for obj_config in obj_config_list:
        del obj_config[KEY_PARAMS][BOOL_PARAM_NAME]

    parsed_config_list = parse_config_object_list(
        obj_category_name,
        obj_type_name,
        obj_config_list,
        obj_factory,
        parse_event_dispatcher,
        params=override_params
    )

    for obj_config_tuple in parsed_config_list:
        (obj_config, obj_config_dict) = obj_config_tuple
        assert obj_config.params[BOOL_PARAM_NAME] == (not BOOL_PARAM_VALUE), \
            'expected override param default value instead of the original default value'
        assert obj_config_dict[KEY_NAME] == obj_config.name, \
            'expected parsed object name to be the same as in the original object dictionary'
        assert obj_config_dict[KEY_PARAMS].get(BOOL_PARAM_NAME, True), \
            'did not expect the original object dictionary to have the boolean parameter'

    # test exclusion of the unknown object in the parsed list
    obj_config_list.append({KEY_NAME: obj_name_unknown, KEY_PARAMS: {}})
    parsed_config_list = parse_config_object_list(
        obj_category_name,
        obj_type_name,
        obj_config_list,
        obj_factory,
        parse_event_dispatcher
    )
    assert len(parsed_config_list) == len(obj_config_list) - 1, \
        'expected the unknown object to be excluded from the parsed list'
    for (obj_config, _) in parsed_config_list:
        assert obj_config.name != obj_name_unknown, \
            'did not expect the unknown object to be parsed successfully'
