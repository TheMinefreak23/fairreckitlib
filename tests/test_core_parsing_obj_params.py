"""This module tests the object (parameters) parsing functionality.

Functions:

    test_trim_config_params: test trimming of (unused) configuration parameters.
    test_parse_config_param: test parsing of a single configuration parameter.
    test_parse_config_parameters: test parsing of the configuration parameters as a whole.
    test_parse_config_obj: test parsing of an object name and parameters configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import pytest

from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.config.config_factories import Factory
from src.fairreckitlib.core.config.config_parameters import ConfigParameters
from src.fairreckitlib.core.parsing.parse_config_object import parse_config_object
from src.fairreckitlib.core.parsing.parse_config_params import \
    parse_config_parameters, parse_config_param, trim_config_params
from src.fairreckitlib.core.parsing.parse_event import ON_PARSE, print_parse_event

BOOL_PARAM_NAME = 'bool'
INT_PARAM_NAME = 'int'
FLOAT_PARAM_NAME = 'float'
SEED_PARAM_NAME = 'seed'
UNKNOWN_PARAM_NAME = 'unknown'

INVALID_DICT_TYPES = [None, True, False, 0, 0.0, 'a', [], {UNKNOWN_PARAM_NAME: 'dict'}, {'set'}]


def create_dummy_config_params() -> ConfigParameters:
    """Create dummy configuration parameters to use for testing."""
    config_params = ConfigParameters()
    config_params.add_random_seed(SEED_PARAM_NAME)
    config_params.add_bool(BOOL_PARAM_NAME, True)
    config_params.add_number(INT_PARAM_NAME, int, 10, (0, 100))
    config_params.add_number(FLOAT_PARAM_NAME, float, 10.0, (0.0, 100.0))
    config_params.add_range('int_range', int, (1, 2), (0, 3))
    config_params.add_range('float_range', float, (1.0, 2.0), (0.0, 3.0))
    config_params.add_single_option('single', str, 'a', ['a', 'b', 'c'])
    config_params.add_multi_option('multi', ['a', 'c'], ['a', 'b', 'c'])
    return config_params


def test_trim_config_params() -> None:
    """Test config parameter trimming with and without an unknown parameter."""
    event_dispatcher = EventDispatcher()
    config_params = create_dummy_config_params()
    param_defaults = config_params.get_defaults()

    # test trimming without an unknown parameter.
    trimmed_params = trim_config_params(param_defaults, 'test', config_params, event_dispatcher)

    assert len(trimmed_params) == len(param_defaults), 'did not expect parameters to be trimmed.'
    for param_name in config_params.get_param_names():
        assert param_name in trimmed_params, 'expected original parameter to be present.'

    # test trimming with an unknown parameter.
    param_defaults[UNKNOWN_PARAM_NAME] = False
    trimmed_params = trim_config_params(param_defaults, 'test', config_params, event_dispatcher)

    assert len(trimmed_params) + 1 == len(param_defaults), 'expected parameters to be trimmed.'
    for param_name in config_params.get_param_names():
        assert param_name in trimmed_params, 'expected original parameter to be present.'
    with pytest.raises(KeyError):
        # test failure to retrieve unknown param after trimming
        _ = trimmed_params[UNKNOWN_PARAM_NAME]


def test_parse_config_param() -> None:
    """Test parsing of a single configuration parameter."""
    event_dispatcher = EventDispatcher()
    event_dispatcher.add_listener(ON_PARSE, None, (lambda _, args: print_parse_event(args), None))
    config_params = create_dummy_config_params()

    for param_name in config_params.get_param_names():
        params = config_params.get_defaults()
        config_param = config_params.get_param(param_name)

        # test success for happy path (using the default value)
        success, parsed_value = parse_config_param(params, '', config_param, event_dispatcher)
        assert success and parsed_value == config_param.default_value, \
            'expected parameter default to succeed when parsing.'

        # test success for int/float casting,
        # for simplicity only primitives as casting is already tested in config param validation
        cast_switch = {INT_PARAM_NAME: float, FLOAT_PARAM_NAME: int}
        if param_name in cast_switch:
            params[param_name] = cast_switch[param_name](params[param_name])
            assert parse_config_param(params, 'Casting', config_param, event_dispatcher)[0], \
                'expected casting between int/floats to succeed when parsing.'

        # test failure for invalid param value,
        # None is not accepted for any config param class, except for a random seed param
        params[param_name] = -1 if param_name == SEED_PARAM_NAME else None
        success, parsed_val = parse_config_param(params, '', config_param, event_dispatcher)
        assert not success, 'expected invalid parameter value to fail on parsing.'
        assert parsed_val == (0 if param_name == SEED_PARAM_NAME else config_param.default_value),\
            'expected invalid parameter value to fail on parsing.'

        # test failure for missing param value
        del params[param_name]
        assert not parse_config_param(
            params,
            'param \'' + param_name + '\' was removed =>',
            config_param,
            event_dispatcher
        )[0], 'expected missing parameter to fail on parsing.'


def test_parse_config_parameters() -> None:
    """Test parsing of the configuration parameters as a whole."""
    event_dispatcher = EventDispatcher()
    event_dispatcher.add_listener(ON_PARSE, None, (lambda _, args: print_parse_event(args), None))
    config_params = create_dummy_config_params()

    # test failure for parsing various types, including a dict that will be empty after trimming
    for params in INVALID_DICT_TYPES:
        params = parse_config_parameters(params, str(params), config_params, event_dispatcher)
        assert params == config_params.get_defaults(), \
            'expected config parameters parsing to fail and return the defaults'

    # test success on updated param values,
    # using boolean for simplicity as the individual param parsing/validations are already tested
    params = config_params.get_defaults()
    default_bool_param = params[BOOL_PARAM_NAME]
    params[BOOL_PARAM_NAME] = not default_bool_param
    params = parse_config_parameters(params, str(params), config_params, event_dispatcher)
    assert params != config_params.get_defaults(), \
        'expected config parameters parsing to succeed and not be the same as the defaults.'
    assert params[BOOL_PARAM_NAME] == (not default_bool_param), \
        'expected the boolean config parameter value to be flipped.'


def test_parse_config_obj() -> None:
    """Test parsing of an object name and parameters configuration."""
    event_dispatcher = EventDispatcher()
    event_dispatcher.add_listener(ON_PARSE, None, (lambda _, args: print_parse_event(args), None))

    def create_dummy_obj(name: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Create dummy object function not used, only to initialize the object factory."""
        return name, params

    obj_type_name = 'test_obj'
    obj_name_unknown = 'unknown_obj'
    obj_name = 'obj'
    obj_factory = Factory('obj_factory')
    obj_factory.add_obj(obj_name, create_dummy_obj, create_dummy_config_params)

    # test failure on incorrect types, including an invalid dictionary
    for obj_config in INVALID_DICT_TYPES:
        parsed_config, parsed_obj_name = parse_config_object(
            obj_type_name,
            obj_config,
            obj_factory,
            event_dispatcher
        )
        assert parsed_config is None, 'did not expect parsed obj configuration on failure'
        assert parsed_obj_name is None, 'did not expect parsed obj name on failure'

    # test failure on dictionary with an object name that is not known in the factory
    obj_config = {KEY_NAME: obj_name_unknown}
    parsed_config, parsed_obj_name = parse_config_object(
        obj_type_name,
        obj_config,
        obj_factory,
        event_dispatcher
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
            event_dispatcher
        )
        assert parsed_obj_name == obj_name, \
            'expected parsed obj name to be the same as the known object name'
        assert parsed_config.name == obj_name, \
            'expected the parsed config obj name to be the same as the known object name'
        assert parsed_config.params == obj_param_defaults, \
            'expected the parsed config params to be the same as the defaults'
