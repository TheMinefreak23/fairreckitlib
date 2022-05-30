"""This module contains functionality for configuration option parameters.

Classes:

    ConfigSingleOptionParam: parameter that can be a single value from a known list of options.
    ConfigMultiOptionParam: parameter that can be multiple values from a known list of options.

Functions:

    create_bool_param: create boolean option parameter.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, List, Tuple, Type, Union

from .config_base_param import ConfigOptionParam


class ConfigMultiOptionParam(ConfigOptionParam):
    """Config Multi Option Parameter.

    The default_value and all the options are expected to be strings.
    The default_value list is expected to have at least one entry.
    The default_value list is expected to be present in the list of available options.
    """

    def __init__(
            self,
            name: str,
            default_value: List[str],
            options: List[str]):
        """Construct the ConfigMultiOptionParam.

        Args:
            name: the name of the parameter.
            default_value: the default option(s) of the parameter.
            options: list of available options for the parameter.
        """
        ConfigOptionParam.__init__(self, name, str, default_value, options)

    def validate_value(self, value: Any) -> Tuple[bool, List[str], str]:
        """Validate the specified value with the parameter.

        Checks for None, type mismatch, duplicates and if the value is one of the allowed options.

        Args:
            value: the list of options to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        if value is None:
            return False, self.default_value, \
                   'no value(s) specified, expected one or more of: ' + str(self.options) + '.'

        # type checks
        if not isinstance(value, list):
            return False, self.default_value, 'Expected list got '+ str(type(value)) + '.'

        # list entries check
        if len(value) == 0:
            return False, self.default_value, 'Expected one or more list entries.'

        # split options into correct/incorrect/duplicate lists
        incorrect, correct, duplicates = [], [], []
        for val in value:
            # check val is correct
            if isinstance(val, self.value_type) and val in self.options:
                # check val is duplicate
                (duplicates if val in correct else correct).append(val)
            # check val is incorrect
            else:
                incorrect.append(val)

        # create error message from incorrect/duplicate lists
        error = ''
        if len(duplicates) > 0:
            error = 'Duplicates: ' + str(duplicates) + '. '
        if len(incorrect) > 0:
            error = error + 'Invalidated: ' + str(incorrect) + '. '

        # check if one or more values are correct
        if len(correct) == 0:
            return False, self.default_value, 'Expected at least one correct value. ' + error

        return True, correct, error


class ConfigSingleOptionParam(ConfigOptionParam):
    """Config Single Option Parameter.

    The default_value and all the options are expected to be either strings or booleans.
    The default_value is expected to be present in the list of available options.
    The options list is expected to have unique values.
    """

    def __init__(
            self,
            name: str,
            value_type: Type,
            default_value: Union[str, bool],
            options: Union[List[str], List[bool]]):
        """Construct the ConfigOptionParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default option of the parameter.
            options: list of available options for the parameter.
        """
        ConfigOptionParam.__init__(self, name, value_type, default_value, options)

    def validate_value(self, value: Any) -> Tuple[bool, Union[str, bool], str]:
        """Validate the specified value with the parameter.

        Checks for None, type mismatch and if the value is one of the allowed options.

        Args:
            value: the option to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        if value is None:
            return False, self.default_value, 'No value specified for \'' + \
                   self.name  + '\', expected one of: ' + str(self.options) + '.'

        # type check
        if not isinstance(value, self.value_type):
            return False, self.default_value, \
                   'Expected ' + str(self.value_type) + ' got '+ str(type(value)) + '.'

        # value check
        if value not in self.options:
            return False, self.default_value, 'Expected one of: ' + str(self.options)

        return True, value, ''


def create_bool_param(name: str, default_value: bool) -> ConfigSingleOptionParam:
    """Create a boolean option parameter.

    Args:
        name: the name of the boolean parameter.
        default_value: the default boolean value.

    Returns:
        the boolean parameter.
    """
    return ConfigSingleOptionParam(name, bool, default_value, [True, False])
