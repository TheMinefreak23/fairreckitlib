"""This module contains functionality for base configuration parameters.

Classes:

    ConfigParam: base class for all parameters.
    ConfigOptionParam: (base) parameter that can be a value from a known list of options.
    ConfigValueParam: (base) parameter that can be a value between a minimum and maximum.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

PARAM_KEY_NAME = 'name'
PARAM_KEY_DEFAULT = 'default'
PARAM_KEY_MIN = 'min'
PARAM_KEY_MAX = 'max'
PARAM_KEY_OPTIONS = 'options'
PARAM_KEY_VALUES = 'values'


class ConfigParam(metaclass=ABCMeta):
    """Config Param base class.

    Public methods:

    to_dict
    validate_value
    """

    def __init__(self, name: str, value_type: Type, default_value: Any):
        """Construct the ConfigParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default value of the parameter.
        """
        self.name = name
        self.value_type = value_type
        self.default_value = default_value

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary describing the parameter.

        The dictionary should at least contains keys for the name
        and default value of the parameter.

        Returns:
            the dictionary containing the parameter descriptions.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate_value(self, value: Any) -> Tuple[bool, Any, str]:
        """Validate the specified value with the parameter.

        Derived implementations should at least check for type mismatch and None.

        Args:
            value: the value to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        raise NotImplementedError()


class ConfigOptionParam(ConfigParam, metaclass=ABCMeta):
    """Config Option Parameter.

    The default_value and all the options are expected to be of the same value_type.
    The default_value is expected to be present in the list of available options.
    """

    def __init__(
            self,
            name: str,
            value_type: Type,
            default_value: Any,
            options: List[Any]):
        """Construct the ConfigOptionParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default option of the parameter.
            options: list of available options for the parameter.
        """
        ConfigParam.__init__(self, name, value_type, default_value)
        self.options = options

    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary describing the option parameter.

        The dictionary contains keys for the name, default value
        and options of the parameter.

        Returns:
            the dictionary containing the parameter descriptions.
        """
        return {
            PARAM_KEY_NAME: self.name,
            PARAM_KEY_OPTIONS: self.options,
            PARAM_KEY_DEFAULT: self.default_value
        }


class ConfigValueParam(ConfigParam, metaclass=ABCMeta):
    """Config Value Parameter.

    The value_type of default_value and min_max_value types are all expected to be either
    int or float, conversions between the two during validation is available.
    The default_value is expected to be between the min_max_value.
    The min_max_value is expected to have min_value <= max_value.
    """

    def __init__(
            self,
            name: str,
            value_type: Type,
            default_value: Any,
            min_max_value: Union[Tuple[int, int], Tuple[float, float]]):
        """Construct the ConfigValueParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default value of the parameter.
            min_max_value: tuple with the minimum and maximum value of the parameter.
        """
        ConfigParam.__init__(self, name, value_type, default_value)
        self.min_value = min_max_value[0]
        self.max_value = min_max_value[1]

    def to_dict(self) -> Dict[str, Any]:
        """Get a dictionary describing the value parameter.

        The dictionary contains keys for the name, default value,
        minimum value and maximum value of the parameter.

        Returns:
            the dictionary containing the parameter descriptions.
        """
        return {
            PARAM_KEY_NAME: self.name,
            PARAM_KEY_MIN: self.min_value,
            PARAM_KEY_MAX: self.max_value,
            PARAM_KEY_DEFAULT: self.default_value
        }
