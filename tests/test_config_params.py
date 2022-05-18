"""This module tests the functionality of configuration param(s).

Functions:

    test_config_option_param: test ConfigOptionParam.
    test_config_value_param: test ConfigValueParam.
    test_config_random_param: test ConfigRandomParam.
    test_config_parameters: test ConfigParameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pytest

from src.fairreckitlib.core.config_params import ConfigOptionParam
from src.fairreckitlib.core.config_params import ConfigRandomParam
from src.fairreckitlib.core.config_params import ConfigValueParam
from src.fairreckitlib.core.config_params import create_bool_param
from src.fairreckitlib.core.config_params import create_empty_parameters


@pytest.mark.parametrize('value_type, valid, invalid', [
    (str, ['a', 'b', 'c'], [None, True, False, 1, 1.0]),
    (bool, [True, False], [None, 'a', 1, 1.0])
])
def test_config_option_param(value_type, valid, invalid):
    """Tests the validation of a ConfigOptionParam.

    Args:
        value_type(type): type of the parameter.
        valid(array like): list of valid options.
        invalid(array like): list of invalid options.
    """
    param = ConfigOptionParam(
        'option_param',
        value_type,
        valid[0],
        valid
    )

    # test all valid options
    for _, opt in enumerate(valid):
        success, value, msg = param.validate_value(opt)
        assert success, 'Option param validation should succeed.'
        assert value == opt, 'Input value should be the same as the output value.'
        if len(msg) > 0:
            print('ConfigOptionParam valid:', msg)

    # test all invalid options
    for _, opt in enumerate(invalid):
        success, value, msg = param.validate_value(opt)
        assert not success, 'Option param validation should not succeed.'
        assert value == param.default_value, 'Validated value should resolve to default.'
        assert len(msg) > 0, 'No error message specified on param value invalidation.'
        print('ConfigOptionParam invalid:', msg)


@pytest.mark.parametrize('value_type, default_value, min_max_value, valid, invalid', [
    (int, 1, (0, 2), [0, 1, 2, 0.1, 2.1], [None, True, False, 'a', -1, 3, -1.1, 3.1]),
    (float, 0.5, (0.0, 1.0), [0.0, 1.0, 0, 1], [None, True, False, 'a', -1, 2, -0.1, 1.1])
])
def test_config_value_param(value_type, default_value, min_max_value, valid, invalid):
    """Tests the validation of a ConfigValueParam.

    Args:
        value_type(int/float): type of the parameter.
        default_value(int/float): default value of the parameter.
        min_max_value(int/float): tuple with the minimum and maximum value of the parameter.
        valid(array like): list of valid values.
        invalid(array like): list of invalid values.
    """
    param = ConfigValueParam(
        'value_param',
        value_type,
        default_value,
        min_max_value
    )

    # test all valid values
    for _, opt in enumerate(valid):
        success, value, msg = param.validate_value(opt)
        assert success, 'Value param validation should succeed.'
        assert value == value_type(opt), 'Input value should be the same as the output value.'
        if len(msg) > 0:
            print('ConfigValueParam valid:', msg)

    # test all invalid values
    for _, opt in enumerate(invalid):
        success, value, msg = param.validate_value(opt)
        assert not success, 'Value param validation should not succeed.'
        assert len(msg) > 0, 'No error message specified on param value invalidation.'
        print('ConfigValueParam invalid:', msg)

        # bool needs to be handled separately, because python considers it to be an int
        if (isinstance(opt, (float, int))) and not isinstance(opt, bool):
            if opt < param.min_value:
                assert value == param.min_value, 'Validated value should be clamped to minimum.'
            elif opt > param.max_value:
                assert value == param.max_value, 'Validated value should be clamped to maximum.'
            else:
                assert value == opt, 'Validated value should not be clamped.'
        else:
            assert value == param.default_value, 'Validated value should resolve to default.'


def test_config_random_param():
    """Tests the validation of a ConfigRandomParam."""
    success, value, _ = ConfigRandomParam('test_param').validate_value(None)
    assert success, 'Random param validation should succeed on None.'
    assert value is None, 'Validated value should still be None.'


def test_config_parameters():
    """Tests the ConfigParameters container.

    1) Add all different types of parameters and check internal counters.
    2) Repeat step 1 but check the opposite, because duplicates are not allowed.
    3) Check correctness of parameter defaults.
    """
    config_parameters = create_empty_parameters()
    assert config_parameters.get_num_params() == 0, 'Empty parameters should have no entries.'

    param_list = [
        create_bool_param('bool', True),
        ConfigOptionParam('option', str, 'a', ['a', 'b', 'c']),
        ConfigRandomParam('random'),
        ConfigValueParam('int', int, 1, (0, 2)),
        ConfigValueParam('float', float, 0.5, (0.0, 1.0))
    ]

    def add_param(param, param_exists):
        if param_exists:
            assert config_parameters.get_param(param.name) is not None, 'Param should exist.'
        else:
            assert config_parameters.get_param(param.name) is None, 'Param should not exist.'

        num_options = len(config_parameters.options)
        num_values = len(config_parameters.values)

        if isinstance(param, ConfigOptionParam):
            if isinstance(param.value_type, bool):
                config_parameters.add_bool(
                    param.name,
                    param.default_value
                )
            else:
                config_parameters.add_option(
                    param.name,
                    param.value_type,
                    param.default_value,
                    param.options
                )

            num_options += 1

        elif isinstance(param, ConfigValueParam):
            if isinstance(param, ConfigRandomParam):
                config_parameters.add_random_seed(param.name)
            else:
                config_parameters.add_value(
                    param.name,
                    param.value_type,
                    param.default_value,
                    (param.min_value, param.max_value)
                )

            num_values += 1

        else:
            raise NotImplementedError('Unknown derived config param class.')

        assert param.name in config_parameters.get_param_names(), \
            'Param name should be available after adding.'
        assert config_parameters.get_num_params() == num_options + num_values, \
            'Number of params should the same as the amount of options and values together.'
        assert len(config_parameters.options) == num_options, \
            'Number of option params should be the same as the amount of added options.'
        assert len(config_parameters.values) == num_values, \
            'Number of value params should be the same as the amount of added values.'


    # first pass: add everything to parameters successfully
    for _, config_param in enumerate(param_list):
        add_param(config_param, False)

    # second pass: add everything to parameters again, but unsuccessfully
    for _, config_param in enumerate(param_list):
        num_params = config_parameters.get_num_params()
        num_options = len(config_parameters.options)
        num_values = len(config_parameters.values)

        pytest.raises(KeyError, add_param, config_param, True)

        assert config_parameters.get_num_params() == num_params, \
            'Number of params should be unchanged.'
        assert len(config_parameters.options) == num_options, \
            'Number of option params should be unchanged.'
        assert len(config_parameters.values) == num_values, \
            'Number of value params should be unchanged.'

    # check parameter defaults
    param_defaults = config_parameters.get_defaults()
    for _, config_param in enumerate(param_list):
        assert config_param.name in param_defaults, \
            'Original param name should be present in the defaults.'
        assert config_param.default_value == param_defaults[config_param.name], \
            'Param default should be the same as the original default value.'
