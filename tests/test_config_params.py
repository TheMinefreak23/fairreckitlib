"""This module tests the functionality of configuration param(s).

Functions:

    test_config_multi_option_param: test ConfigMultiOptionParam.
    test_config_single_option_param: test ConfigSingleOptionParam.
    test_config_number_param: test ConfigNumberParam.
    test_config_random_param: test ConfigRandomParam.
    test_config_range_param: test ConfigRangeParam.
    test_config_parameters: test ConfigParameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, List, Tuple, Type, Union

import pytest

from src.fairreckitlib.core.params.config_base_param import ConfigParam,PARAM_KEY_MIN,PARAM_KEY_MAX
from src.fairreckitlib.core.params.config_value_param import ConfigValueParam, \
    ConfigNumberParam, ConfigRandomParam, ConfigRangeParam
from src.fairreckitlib.core.params.config_option_param import ConfigOptionParam, \
    ConfigMultiOptionParam, ConfigSingleOptionParam, create_bool_param
from src.fairreckitlib.core.params.config_parameters import create_empty_parameters


def test_config_multi_option_param() -> None:
    """Test the validation of a ConfigMultiOptionParam."""
    def assert_correct_multi_option(param: ConfigMultiOptionParam, value: List[str]) -> None:
        """Assert the multi option param validation be correct.

        Args:
            param: the option param to use for validation.
            value: the value to use for validation.
        """
        success, ret_value, msg = param.validate_value(value)
        assert success, 'Expected correct multi option value validation.'
        assert len(msg) == 0, 'Did not expect msg for correct multi option value validation.'
        assert ret_value == value, 'Expected return value to be the same as the input value.'
        print('ConfigMultiOptionParam valid: Value', value, 'in', param.options)

    def assert_truncated_multi_option(param: ConfigMultiOptionParam, value: List[str]) -> None:
        """Assert the multi option param validation be correct, but truncated.

        Meaning the value contains either duplicates or invalid values,
        but at least one valid entry is present.

        Args:
            param: the option param to use for validation.
            value: the value to use for validation.
        """
        success, ret_value, msg = param.validate_value(value)
        assert success, 'Expected correctly truncated multi option value validation.'
        assert len(msg) > 0, 'Expected msg for correctly truncated multi option value validation.'
        assert len(ret_value) < len(value), 'Expected truncated result to be less than input value.'
        print('ConfigMultiOptionParam valid: Value', ret_value, 'truncated from', value, '->', msg)

    def assert_incorrect_multi_option(param: ConfigMultiOptionParam, value: Any) -> None:
        """Assert the multi option param validation be incorrect.

        Args:
            param: the option param to use for invalidation.
            value: the value to use for invalidation.
        """
        success, ret_value, msg = param.validate_value(value)
        assert not success, 'Expected incorrect multi option value validation.'
        assert len(msg) > 0, 'Expected msg for incorrect multi option validation.'
        assert ret_value != value, 'Expected ret_value to differ on incorrect input.'
        assert ret_value == param.default_value, 'Expected ret_value to be default when incorrect.'
        print('ConfigMultiOptionParam invalid:', msg)

    valid = ['a', 'b']
    invalid = [None, True, False, 1, 1.0]
    multi_option = ConfigMultiOptionParam(
        'multi_option_param',
        valid,
        valid
    )

    # test success when the supplied value is the same as all available options
    assert_correct_multi_option(multi_option, valid)

    # test failure when the supplied value is an empty list
    assert_incorrect_multi_option(multi_option, [])

    for invalid_entry in invalid:
        # test failure when the supplied value is not a list
        assert_incorrect_multi_option(multi_option, invalid_entry)

    for valid_entry in valid:
        # test individual values to be valid
        assert_correct_multi_option(multi_option, [valid_entry])

        for i, i_entry in enumerate(valid):
            for j, j_entry in enumerate(valid):
                entry = [i_entry, j_entry]
                multi_option = ConfigMultiOptionParam(
                    'multi_option_param',
                    entry,
                    valid
                )
                if i == j:
                    # test duplicate valid entries
                    assert_truncated_multi_option(multi_option, entry)
                else:
                    # test various combos of valid entries
                    assert_correct_multi_option(multi_option, entry)

        for i_entry in invalid:
            multi_option = ConfigMultiOptionParam(
                'multi_option_param',
                valid,
                valid
            )

            # test individual values to be invalid
            assert_incorrect_multi_option(multi_option, [i_entry])
            # test still valid with at least one valid entry
            assert_truncated_multi_option(multi_option, [valid_entry, i_entry])

            for j_entry in invalid:
                # test various combos of invalid entries (including duplicates)
                assert_incorrect_multi_option(multi_option, [i_entry, j_entry])
                # test still valid with at least one valid entry (order should not matter)
                assert_truncated_multi_option(multi_option, [i_entry, valid_entry, j_entry])


@pytest.mark.parametrize('value_type, valid, invalid', [
    (str, ['a', 'b', 'c'], [None, True, False, 1, 1.0]),
    (bool, [True, False], [None, 'a', 1, 1.0])
])
def test_config_single_option_param(
        value_type: Type,
        valid: List[Union[str, bool]],
        invalid: List[Any]) -> None:
    """Test the validation of a ConfigSingleOptionParam.

    Args:
        value_type: type of the parameter.
        valid: list of valid options.
        invalid: list of invalid options.
    """
    param = ConfigSingleOptionParam(
        'single_option_param',
        value_type,
        valid[0],
        valid
    )

    # test all valid options
    for opt in valid:
        success, value, msg = param.validate_value(opt)
        assert success, 'Option param validation should succeed.'
        assert value == opt, 'Input value should be the same as the output value.'
        if len(msg) > 0:
            print('ConfigSingleOptionParam valid:', msg)

    # test all invalid options
    for opt in invalid:
        success, value, msg = param.validate_value(opt)
        assert not success, 'Option param validation should not succeed.'
        assert value == param.default_value, 'Validated value should resolve to default.'
        assert len(msg) > 0, 'No error msg specified on param value invalidation.'
        print('ConfigSingleOptionParam invalid:', msg)


@pytest.mark.parametrize('value_type, default_value, min_max_value, valid, invalid', [
    (int, 1, (0, 2), [0, 1, 2, 0.1, 2.1], [None, True, False, 'a', -1, 3, -1.1, 3.1]),
    (float, 0.5, (0.0, 1.0), [0.0, 1.0, 0, 1], [None, True, False, 'a', -1, 2, -0.1, 1.1])
])
def test_config_number_param(
        value_type: Type,
        default_value: Union[int, float],
        min_max_value: Union[Tuple[int, int], Tuple[float, float]],
        valid: List[Union[int, float]],
        invalid: List[Any]) -> None:
    """Test the validation of a ConfigNumberParam.

    Args:
        value_type: type of the parameter.
        default_value: default value of the parameter.
        min_max_value: tuple with the minimum and maximum value of the parameter.
        valid: list of valid values.
        invalid: list of invalid values.
    """
    param = ConfigNumberParam(
        'number_param',
        value_type,
        default_value,
        min_max_value
    )

    # test all valid values
    for opt in valid:
        success, value, msg = param.validate_value(opt)
        assert success, 'Value param validation should succeed.'
        assert value == value_type(opt), 'Input value should be the same as the output value.'
        if len(msg) > 0:
            print('ConfigNumberParam valid:', msg)

    # test all invalid values
    for opt in invalid:
        success, value, msg = param.validate_value(opt)
        assert not success, 'Value param validation should not succeed.'
        assert len(msg) > 0, 'No error msg specified on param value invalidation.'
        print('ConfigNumberParam invalid:', msg)

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


def test_config_random_param() -> None:
    """Test the validation of a ConfigRandomParam."""
    success, value, _ = ConfigRandomParam('random_seed').validate_value(None)
    assert success, 'Random param validation should succeed on None.'
    assert value is None, 'Validated value should still be None.'


@pytest.mark.parametrize('value_type, default_value, min_value, max_value', [
    (int, (1, 2), 0, 3),
    (float, (1.0, 2.0), 0.0, 3.0)
])
def test_config_range_param(
        value_type: Type,
        default_value: Union[Tuple[int, int], Tuple[float, float]],
        min_value: Union[int, float],
        max_value: Union[int, float]) -> None:
    """Test the validation of a ConfigRangeParam.

    Args:
        value_type: type of the parameter.
        default_value: default value of the parameter.
        min_value: the minimum value of the parameter.
        max_value: the maximum value of the parameter.
    """
    def assert_correct_range(param: ConfigRangeParam, value: Any) -> None:
        """Assert the range param validation be incorrect.

        Args:
            param: the range param to use for validation.
            value: the value to use for validation.
        """
        success, ret_value, msg = param.validate_value(value)
        assert success, 'Expected correct range value validation.'
        assert len(msg) == 0, 'Did not expect msg for correct range validation.'
        assert ret_value == value, 'Expected ret_value to be the same on correct input.'

    def assert_incorrect_range(param: ConfigRangeParam, value: Any) -> None:
        """Assert the range param validation be incorrect.

        Args:
            param: the range param to use for invalidation.
            value: the value to use for invalidation.
        """
        success, ret_value, msg = param.validate_value(value)
        assert not success, 'Expected incorrect range value validation.'
        assert len(msg) > 0, 'Expected msg for incorrect range validation.'
        assert ret_value != value, 'Expected ret_value to differ on incorrect input.'
        assert ret_value == param.default_value, 'Expected ret_value to be default when incorrect.'
        print('ConfigRangeParam invalid:', msg)

    range_param = ConfigRangeParam(
        'range_param',
        value_type,
        default_value,
        (min_value, max_value)
    )

    valid_dict = [
        {PARAM_KEY_MIN: default_value[0], PARAM_KEY_MAX: default_value[1]},
        {PARAM_KEY_MIN: default_value[0], PARAM_KEY_MAX: max_value},
        {PARAM_KEY_MIN: min_value, PARAM_KEY_MAX: default_value[1]},
        {PARAM_KEY_MIN: min_value, PARAM_KEY_MAX: max_value},
    ]

    invalid_values = [None, [], {}]
    invalid_dict = [
        # test missing PARAM_KEY_MIN or PARAM_KEY_MAX (cast to float when int)
        {PARAM_KEY_MIN: 0}, {PARAM_KEY_MAX: 0},
        # test missing PARAM_KEY_MIN or PARAM_KEY_MAX (cast to int when float)
        {PARAM_KEY_MIN: 0.0}, {PARAM_KEY_MAX: 0.0},
        # test boundary min and max value (cast to float when int)
        {PARAM_KEY_MIN: min_value - 1, PARAM_KEY_MAX: max_value + 1},
        # test boundary min and max value (cast to int when float)
        {PARAM_KEY_MIN: min_value - 1.0, PARAM_KEY_MAX: max_value + 1.0},
        # test swapping min and max
        {PARAM_KEY_MIN: max_value, PARAM_KEY_MAX: min_value}
    ]

    for val in valid_dict:
        # test success on correct dict input
        assert_correct_range(range_param, val)

    for val in invalid_values:
        # test failure on incorrect container input
        assert_incorrect_range(range_param, val)

        for dict_val in invalid_dict:
            # test failure on incorrect dict input
            assert_incorrect_range(range_param, dict_val)
            # test failure on incorrect dict with invalid PARAM_KEY_MIN input
            assert_incorrect_range(range_param, {PARAM_KEY_MIN: val, PARAM_KEY_MAX: max_value})
            # test failure on incorrect dict with invalid PARAM_KEY_MAX input
            assert_incorrect_range(range_param, {PARAM_KEY_MIN: min_value, PARAM_KEY_MAX: val})


def test_config_parameters() -> None:
    """Test the ConfigParameters container.

    1) Add all different types of parameters and check internal counters.
    2) Repeat step 1 but check the opposite, because duplicates are not allowed.
    3) Check correctness of parameter defaults.
    """
    config_parameters = create_empty_parameters()
    assert config_parameters.get_num_params() == 0, 'Empty parameters should have no entries.'

    param_list = [
        # option params
        create_bool_param('bool_option', True),
        ConfigMultiOptionParam('multi_option', ['a'], ['a', 'b', 'c']),
        ConfigSingleOptionParam('single_option', str, 'a', ['a', 'b', 'c']),
        # value params
        ConfigNumberParam('int_number', int, 1, (0, 2)),
        ConfigNumberParam('float_number', float, 0.5, (0.0, 1.0)),
        ConfigRandomParam('random_seed'),
        ConfigRangeParam('int_range', int, (1, 1), (0, 2)),
        ConfigRangeParam('float_range', float, (0.5, 0.5), (0.0, 1.0)),
    ]

    def add_option_param(param: ConfigOptionParam) -> None:
        """Add an option parameter to config_parameters.

        Args:
            param: the config parameter to add.
        """
        if isinstance(param.value_type, bool):
            config_parameters.add_bool(
                param.name,
                param.default_value
            )
        elif isinstance(param, ConfigSingleOptionParam):
            config_parameters.add_single_option(
                param.name,
                param.value_type,
                param.default_value,
                param.options
            )
        elif isinstance(param, ConfigMultiOptionParam):
            config_parameters.add_multi_option(
                param.name,
                param.default_value,
                param.options
            )
        else:
            raise NotImplementedError('Unknown derived option param class.')

    def add_value_param(param: ConfigValueParam) -> None:
        """Add a value parameter to config_parameters.

        Args:
            param: the config parameter to add.
        """
        if isinstance(param, ConfigRandomParam):
            config_parameters.add_random_seed(param.name)
        elif isinstance(param, ConfigNumberParam):
            config_parameters.add_number(
                param.name,
                param.value_type,
                param.default_value,
                (param.min_value, param.max_value)
            )
        elif isinstance(param, ConfigRangeParam):
            config_parameters.add_range(
                param.name,
                param.value_type,
                (param.default_value[PARAM_KEY_MIN], param.default_value[PARAM_KEY_MAX]),
                (param.min_value, param.max_value)
            )
        else:
            raise NotImplementedError('Unknown derived value param class.')

    def add_param(param: ConfigParam, param_exists: bool) -> None:
        """Add a parameter to config_parameters.

        Args:
            param: the config parameter to add.
            param_exists: whether the param should exist in config_parameters.
        """
        if param_exists:
            assert config_parameters.get_param(param.name) is not None, 'Param should exist.'
        else:
            assert config_parameters.get_param(param.name) is None, 'Param should not exist.'

        num_opts = len(config_parameters.options)
        num_vals = len(config_parameters.values)

        if isinstance(param, ConfigOptionParam):
            add_option_param(param)
            num_opts += 1
        elif isinstance(param, ConfigValueParam):
            add_value_param(param)
            num_vals += 1
        else:
            raise NotImplementedError('Unknown derived config param class.')

        assert param.name in config_parameters.get_param_names(), \
            'Param name should be available after adding.'
        assert config_parameters.get_num_params() == num_opts + num_vals, \
            'Number of params should the same as the amount of options and values together.'
        assert len(config_parameters.options) == num_opts, \
            'Number of option params should be the same as the amount of added options.'
        assert len(config_parameters.values) == num_vals, \
            'Number of value params should be the same as the amount of added values.'


    # first pass: add everything to parameters successfully
    for config_param in param_list:
        add_param(config_param, False)

    # second pass: add everything to parameters again, but unsuccessfully
    for config_param in param_list:
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
    for config_param in param_list:
        assert config_param.name in param_defaults, \
            'Original param name should be present in the defaults.'
        assert config_param.default_value == param_defaults[config_param.name], \
            'Param default should be the same as the original default value.'
