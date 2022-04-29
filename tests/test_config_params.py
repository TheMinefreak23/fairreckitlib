"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pytest

from src.fairreckitlib.experiment.params import ConfigOptionParam
from src.fairreckitlib.experiment.params import ConfigRandomParam
from src.fairreckitlib.experiment.params import ConfigValueParam
from src.fairreckitlib.experiment.params import create_bool_param
from src.fairreckitlib.experiment.params import get_empty_parameters


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
        if (isinstance(opt, float) or isinstance(opt, int)) and not isinstance(opt, bool):
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
    config_parameters = get_empty_parameters()
    assert config_parameters.get_num_params() == 0, 'Empty parameters should have no entries.'

    params = [
        create_bool_param('bool', True),
        ConfigOptionParam('option', str, 'a', ['a', 'b', 'c']),
        ConfigRandomParam('random'),
        ConfigValueParam('int', int, 1, (0, 2)),
        ConfigValueParam('float', float, 0.5, (0.0, 1.0))
    ]

    num_options = 0
    num_values = 0

    # first pass: add everything to parameters successfully
    for _, p in enumerate(params):
        assert config_parameters.get_param(p.name) is None, 'Param should not exist.'

        if isinstance(p, ConfigOptionParam):
            if isinstance(p.value_type, bool):
                config_parameters.add_bool(
                    p.name,
                    p.default_value
                )
            else:
                config_parameters.add_option(
                    p.name,
                    p.value_type,
                    p.default_value,
                    p.options
                )

            num_options += 1

        elif isinstance(p, ConfigValueParam):
            if isinstance(p, ConfigRandomParam):
                config_parameters.add_random_seed(
                    p.name
                )
            else:
                config_parameters.add_value(
                    p.name,
                    p.value_type,
                    p.default_value,
                    (p.min_value, p.max_value)
                )

            num_values += 1

        else:
            raise NotImplementedError()

        assert p.name in config_parameters.get_param_names(), \
            'Param name should be available after adding.'
        assert config_parameters.get_num_params() == num_options + num_values, \
            'Number of params should the same as the amount of options and values together.'
        assert len(config_parameters.options) == num_options, \
            'Number of options params should be the same as the amount of added options.'
        assert len(config_parameters.values) == num_values, \
            'Number of value params should be the same as the amount of added values.'

    # second pass: add everything to parameters again, unsuccessfully
    for _, p in enumerate(params):
        assert config_parameters.get_param(p.name) is not None, 'Param should exist.'

        error_raised = False

        try:
            if isinstance(p, ConfigOptionParam):
                if isinstance(p.value_type, bool):
                    config_parameters.add_bool(
                        p.name,
                        p.default_value
                    )
                else:
                    config_parameters.add_option(
                        p.name,
                        p.value_type,
                        p.default_value,
                        p.options
                    )
            elif isinstance(p, ConfigValueParam):
                num_values += 1

                if isinstance(p, ConfigRandomParam):
                    config_parameters.add_random_seed(
                        p.name
                    )
                else:
                    config_parameters.add_value(
                        p.name,
                        p.value_type,
                        p.default_value,
                        (p.min_value, p.max_value)
                    )
            else:
                raise NotImplementedError()
        except KeyError:
            error_raised = True
        finally:
            assert error_raised, 'Error should be raised on adding duplicate param name.'

    # check parameter defaults
    param_defaults = config_parameters.get_defaults()
    for _, p in enumerate(params):
        assert p.name in param_defaults, \
            'Original param name should be present in the defaults.'
        assert p.default_value == param_defaults[p.name], \
            'Param default should be the same as the original default value.'
