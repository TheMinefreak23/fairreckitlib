"""This module tests the core parsing functionality.

Functions:

    test_parse_assert_empty_container: test assertion of (non) empty containers.
    test_parse_assert_key_in_dict_and_one_of_list: test assertion of key/value in container.
    test_parse_assert_is_type: test assertion for value to be of a certain type.
    test_trim_config_params: test trimming of (unused) configuration parameters.
    test_parse_config_param: test parsing of a single configuration parameter.
    test_parse_config_parameters: test parsing of the configuration parameters as a whole.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Dict, List, Union

import pytest

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.params.config_parameters import ConfigParameters
from src.fairreckitlib.core.parsing.parse_assert import \
    assert_is_container_not_empty, assert_is_key_in_dict, assert_is_one_of_list, assert_is_type
from src.fairreckitlib.core.parsing.parse_event import ON_PARSE, print_parse_event
from src.fairreckitlib.core.parsing.parse_params import \
    parse_config_parameters, parse_config_param, trim_config_params

BOOL_PARAM_NAME = 'bool'
INT_PARAM_NAME = 'int'
FLOAT_PARAM_NAME = 'float'
SEED_PARAM_NAME = 'seed'
UNKNOWN_PARAM_NAME = 'unknown'

config_params = ConfigParameters()
config_params.add_random_seed(SEED_PARAM_NAME)
config_params.add_bool(BOOL_PARAM_NAME, True)
config_params.add_number(INT_PARAM_NAME, int, 10, (0, 100))
config_params.add_number(FLOAT_PARAM_NAME, float, 10.0, (0.0, 100.0))
config_params.add_range('int_range', int, (1, 2), (0, 3))
config_params.add_range('float_range', float, (1.0, 2.0), (0.0, 3.0))
config_params.add_single_option('single', str, 'a', ['a', 'b', 'c'])
config_params.add_multi_option('multi', ['a', 'c'], ['a', 'b', 'c'])


@pytest.mark.parametrize('container', [{}, [], {'0':0}, [0]])
def test_parse_assert_empty_container(container: Union[Dict, List]) -> None:
    """Test the assertion for (non) empty containers.

    Args:
        container: the container to assert
    """
    event_dispatcher = EventDispatcher()

    if len(container) == 0:
        assert not assert_is_container_not_empty(container, event_dispatcher, ''), \
            'expected container to be empty'
    else:
        assert assert_is_container_not_empty(container, event_dispatcher, ''), \
            'expected container to have entries'


def test_parse_assert_key_in_dict_and_one_of_list() -> None:
    """Test the assertion of a key present in a dictionary or a value present in a list."""
    event_dispatcher = EventDispatcher()

    for i in list(range(0, 100)):
        # keys are assumed to be strings, value can be Any
        key = str(i)
        assert assert_is_key_in_dict(key, {key:i}, event_dispatcher, ''), \
            'expected key to be in the dictionary.'
        assert not assert_is_key_in_dict(key, {}, event_dispatcher, ''), \
            'did not expect key in the dictionary.'

        assert assert_is_one_of_list(i, [i], event_dispatcher, ''), \
            'expected key to be in the list.'
        assert not assert_is_one_of_list(i, [], event_dispatcher, ''), \
            'did not expect key to be in the list.'


def test_parse_assert_is_type() -> None:
    """Test the assertion of various primitive types and containers."""
    event_dispatcher = EventDispatcher()
    types = [None, True, False, 0, 0.0, 'a', [], {'dict':0}, {'set'}]
    for expected_type in types:
        expected_type = type(expected_type)

        for value in types:
            if isinstance(value, expected_type):
                assert assert_is_type(value, expected_type, event_dispatcher, ''), \
                    'expected types to be the same.'
            else:
                assert not assert_is_type(value, expected_type, event_dispatcher, ''), \
                    'expected types to be different.'


def test_trim_config_params() -> None:
    """Test config parameter trimming with and without an unknown parameter."""
    event_dispatcher = EventDispatcher()
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

    # test failure for parsing various types, including a dict that will be empty after trimming
    for params in [None, True, False, 0, 0.0, 'a', [], {UNKNOWN_PARAM_NAME: 'dict'}, {'set'}]:
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
