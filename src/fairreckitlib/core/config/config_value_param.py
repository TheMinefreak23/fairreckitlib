"""This module contains functionality for configuration value parameters.

Classes:

    ConfigNumberParam: parameter that can be a number between a minimum and maximum.
    ConfigRandomParam: parameter that can be used to pick the (optional) random seed.
    ConfigRangeParam: parameter that can be a range between a minimum and maximum.

Functions:

    validate_min_max:
    validate_type:

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import sys
from typing import Any, Dict, Optional, Tuple, Type, Union

from .config_base_param import ConfigValueParam, PARAM_KEY_MIN, PARAM_KEY_MAX


class ConfigNumberParam(ConfigValueParam):
    """Config Number Parameter.

    The value_type of default_value and min_max_value types are all expected to be either
    int or float, conversions between the two during validation is available.
    The default_value is expected to be between the min_max_value.
    The min_max_value is expected to have min_value <= max_value.
    """

    def __init__(
            self,
            name: str,
            value_type: Type,
            default_value: Union[int, float],
            min_max_value: Union[Tuple[int, int], Tuple[float, float]]):
        """Construct the ConfigNumberParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default value of the parameter.
            min_max_value: tuple with the minimum and maximum value of the parameter.
        """
        ConfigValueParam.__init__(self, name, value_type, default_value, min_max_value)

    def validate_value(self, value: Any) -> Tuple[bool, Union[int, float], str]:
        """Validate the specified value with the parameter.

        Checks for None, type mismatch (conversion between int and float is allowed) and if
        the value is between the minimum and maximum value of the parameter.

        Args:
            value: the number to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        if value is None:
            return False, self.default_value, 'No value specified for \'' + \
                   self.name  + '\', expected value between ' + \
                   str(self.min_value) + ' and ' + str(self.max_value)

        # type check
        type_success, value, type_error = \
            validate_type(self.name, value, self.value_type, self.default_value)

        # value check
        val_success, value, val_error = \
            validate_min_max(self.name, value, self.min_value, self.max_value)

        return type_success and val_success, value, type_error + val_error


class ConfigRandomParam(ConfigNumberParam):
    """Config Random Parameter."""

    def __init__(self, name: str):
        """Construct the ConfigRandomParam.

        Args:
            name: the name of the random seed parameter.
        """
        ConfigNumberParam.__init__(self, name, int, None, (0, sys.maxsize))

    def validate_value(self, value: Any) -> Tuple[bool, Optional[int], str]:
        """Validate the specified value with the parameter.

        Checks are the same as ConfigNumberParam.validate_value, but None is allowed.

        Args:
            value: the random seed to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        # skips the 'None' error from ConfigNumberParam
        if value is None:
            return True, value, ''

        return ConfigNumberParam.validate_value(self, value)


class ConfigRangeParam(ConfigValueParam):
    """Config Range Parameter.

    The value_type of default_value and min_max_value types are all expected to be either
    int or float, conversions between the two during validation is available.
    The default_value min and max values are expected to be between the min_max_value.
    The default_value and min_max_value are expected to have min_value <= max_value.
    """

    def __init__(
            self,
            name: str,
            value_type: Type,
            default_value: Union[Tuple[int, int], Tuple[float, float]],
            min_max_value: Union[Tuple[int, int], Tuple[float, float]]):
        """Construct the ConfigRangeParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: tuple with the default minimum and maximum value of the parameter.
            min_max_value: tuple with the minimum and maximum value of the parameter.
        """
        ConfigValueParam.__init__(
            self,
            name,
            value_type,
            {
                PARAM_KEY_MIN: default_value[0],
                PARAM_KEY_MAX: default_value[1]
            },
            min_max_value
        )

    def validate_value(self, value: Any) -> Tuple[bool, Dict[str, Union[int, float]], str]:
        """Validate the specified value with the parameter.

        Checks for None, type mismatch (conversion between int and float is allowed) and if
        the min-max range is between the minimum and maximum value of the parameter.

        Args:
            value: the min-max range dictionary to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        if value is None:
            return False, self.default_value, 'No value specified, expected \'' + \
                   PARAM_KEY_MIN + '\' and \'' + PARAM_KEY_MAX + '\' range between ' + \
                   str(self.min_value) + ' and ' + str(self.max_value)

        # check min-max range dictionary
        success, min_max_range, error = self.validate_dict(value)
        # initialize result value with defaults
        result_val = self.default_value

        # check min-max range type
        for key, val in min_max_range.items():
            val_success, val, val_err = validate_type(key, val, self.value_type, result_val[key])
            success = success and val_success
            result_val[key] = val
            error += val_err

        # check min-max range value
        for key, val in dict(result_val).items():
            val_success, val, val_err = validate_min_max(key, val, self.min_value, self.max_value)
            success = success and val_success
            result_val[key] = val
            error += val_err

        # swap when min range > max range
        if result_val[PARAM_KEY_MIN] > result_val[PARAM_KEY_MAX]:
            result_val[PARAM_KEY_MIN], result_val[PARAM_KEY_MAX] = \
            result_val[PARAM_KEY_MAX], result_val[PARAM_KEY_MIN]
            success = False
            error += 'Swapped \'' + PARAM_KEY_MIN + '\' and \'' + PARAM_KEY_MAX + '\'. '

        return success, result_val, error

    def validate_dict(self, value: Any) -> Tuple[bool, Dict[str, Any], str]:
        """Validate the specified value to a min-max range dictionary.

        Args:
            value: the min-max range dictionary to validate.

        Returns:
            whether it was successful, the validated dictionary and (optional) message.
        """
        if not isinstance(value, dict):
            return False, self.default_value, \
                   'Expected dictionary with \'min\' and \'max\' range between ' + \
                   str(self.min_value) + ' and ' + str(self.max_value)

        success = True
        if PARAM_KEY_MIN in value:
            min_error = ''
            min_val = value[PARAM_KEY_MIN]
        else:
            min_val = self.min_value
            min_error = 'Expected \'min\' in dictionary. '
            success = False

        if PARAM_KEY_MAX in value:
            max_error = ''
            max_val = value[PARAM_KEY_MAX]
        else:
            max_val = self.max_value
            max_error = 'Expected \'max\' in dictionary. '
            success = False

        return success, { PARAM_KEY_MIN: min_val, PARAM_KEY_MAX: max_val }, min_error + max_error


def validate_min_max(
        value_name: str,
        value: Union[int, float],
        min_value: Union[int, float],
        max_value: Union[int, float]) -> Tuple[bool, Union[int, float], str]:
    """Validate the value with the specified min- and max-value.

    Args:
        value_name: the name associated with the value.
        value: the value to validate.
        min_value: the minimum value to use for validation.
        max_value: the maximum value to use for validation.

    Returns:
        whether it was successful, the validated value and (optional) message.
    """
    if value < min_value:
        return False, min_value, 'Expected \'' + value_name + \
               '\' greater or equal to ' + str(min_value) + '. '
    if value > max_value:
        return False, max_value, 'Expected \'' + value_name + \
               '\' less or equal to ' + str(max_value) + '. '

    return True, value, ''

def validate_type(
        value_name: str,
        value: Any,
        value_type: Type,
        default_value: Union[int, float]) -> Tuple[bool, Union[int, float], str]:
    """Validate the value to be of the correct type.

    Conversion between int/float is allowed and will return True, but
    with the optional message that the value was cast to the correct value_type.

    Args:
        value_name: the name associated with the value.
        value: the value to validate the type of.
        value_type: the type of the value.
        default_value: the default value to return on invalidation.

    Returns:
        whether it was successful, the validated value and (optional) message.
    """
    if isinstance(value, float) and value_type == int:
        return True, value_type(value), 'Value \'' + value_name + '\' cast to int. '
    if isinstance(value, int) and not isinstance(value, bool) and value_type == float:
        return True, value_type(value), 'Value \'' + value_name + '\' cast to float. '
    if not isinstance(value, value_type) or isinstance(value, bool):
        return False, default_value, 'Expected \'' + value_name + '\' to be ' + \
               str(value_type) + ' got ' + str(type(value)) + '. '

    return True, value, ''
