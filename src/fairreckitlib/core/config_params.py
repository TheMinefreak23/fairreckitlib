"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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


class ConfigOptionParam(ConfigParam):
    """Config Option Parameter.

    The default value and all the options are expected to be of the same value_type.
    Moreover, the default value is assumed to be included in the options list as well.
    """

    def __init__(
            self, name: str,
            value_type: Type,
            default_value: Union[str, bool],
            options: List[Union[str, bool]]):
        """Construct the ConfigOptionParam.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default option of the parameter.
            options: list of available options for the parameter (including the default).
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

    def validate_value(self, value: Union[str, bool]) -> Tuple[bool, Union[str, bool], str]:
        """Validate the specified value with the parameter.

        Checks for None, type mismatch and if the value is one of the
        allowed options.

        Args:
            value: the value to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        if value is None:
            return False, self.default_value, \
                   'no value specified, expected one of: ' + str(self.options)

        # type check
        if not isinstance(value, self.value_type):
            return False, self.default_value, \
                   'expected ' + str(self.value_type) + ' got '+ str(type(value))

        # value check
        if value in self.options:
            return True, value, ''

        return False, self.default_value, 'expected one of: ' + str(self.options)


class ConfigValueParam(ConfigParam):
    """Config Value Parameter.

    The default, min, and max value are all expected to be of the same value_type.
    The value_type is either an integer or floating-point. Conversions between the
    two during validation is available.
    """

    def __init__(
            self,
            name: str,
            value_type: Type,
            default_value: Union[int, float],
            min_max_value: Tuple[Union[int, float], Union[int, float]]):
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

    def validate_value(self, value: Union[int, float]) -> Tuple[bool, Union[int, float], str]:
        """Validate the specified value with the parameter.

        Checks for None, type mismatch (conversion between int and float is allowed) and if
        the value is between the minimum and maximum value of the parameter.

        Args:
            value: the value to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        if value is None:
            return False, self.default_value, \
                   'no value specified, expected value between ' + \
                   str(self.min_value) + ' and ' + str(self.max_value)

        error = ''
        # type checks
        if isinstance(value, float) and self.value_type == int:
            value = self.value_type(value)
            error = '(value cast to int) '
        elif isinstance(value, int) and not isinstance(value, bool) and self.value_type == float:
            value = self.value_type(value)
            error = '(value cast to float) '
        elif not isinstance(value, self.value_type) or isinstance(value, bool):
            return False, self.default_value, \
                   'expected ' + str(self.value_type) + ' got '+ str(type(value))

        # value checks
        if value < self.min_value:
            return False, self.min_value, error + \
                   'expected value greater or equal to ' + str(self.min_value)
        if value > self.max_value:
            return False, self.max_value, error + \
                   'expected value less or equal to ' + str(self.max_value)

        return True, value, error


class ConfigRandomParam(ConfigValueParam):
    """Config Random Parameter.

    Derived from an integer ConfigValueParam, and in addition
    allows the default_value to be None.
    """

    def __init__(self, name: str):
        """Construct the ConfigRandomParam.

        Args:
            name: the name of the random seed parameter.
        """
        ConfigValueParam.__init__(self, name, int, None, (0, sys.maxsize))

    def validate_value(self, value: Optional[int]) -> Tuple[bool, Optional[int], str]:
        """Validate the specified value with the parameter.

        Checks are the same as ConfigValueParam.validate_value, but None is allowed.

        Args:
            value: the value to verify with the parameter.

        Returns:
            whether it was successful, the validated value and (optional) message.
        """
        # skips the 'None' error from ConfigValueParam
        if value is None:
            return True, value, ''

        return ConfigValueParam.validate_value(self, value)


class ConfigParameters:
    """Config Parameters.

    Container with varying Config parameters using a dictionary.
    Moreover, he added option/value parameters are stored separately.

    Public methods:

    add_bool
    add_option
    add_random_seed
    add_value
    get_defaults
    get_num_params
    get_param
    get_param_names
    to_dict
    """

    def __init__(self):
        """Construct the ConfigParameters."""
        self.params = {}
        self.options = []
        self.values = []

    def add_bool(self, name: str, default_value: bool) -> None:
        """Add a boolean parameter.

        Sugar for an option parameter that is either True or False.
        Raises a KeyError when the name of the parameter is already present.

        Args:
              name: name of the boolean parameter.
              default_value: the default boolean value.
        """
        param = create_bool_param(name, default_value)

        self.add_param(param)
        self.options.append(param)

    def add_option(
            self,
            name: str,
            value_type: Type,
            default_option: Union[str, bool],
            options: List[Union[str, bool]]) -> None:
        """Add an option parameter.

        The default value and all the options are expected
        to be of the same value_type.

        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the parameter.
            value_type:  the type of the parameter (either str or bool).
            default_option: default option of the parameter.
            options: list of available options for the parameter.
        """
        param = ConfigOptionParam(
            name,
            value_type,
            default_option,
            options
        )

        self.add_param(param)
        self.options.append(param)

    def add_random_seed(self, name: str) -> None:
        """Add a random seed parameter.

        An integer value parameter, where the default_value is None.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the random seed parameter.
        """
        param = ConfigRandomParam(name)

        self.add_param(param)
        self.values.append(param)

    def add_value(
            self,
            name: str,
            value_type: Type,
            default_value: Union[int, float],
            min_max_value: Tuple[Union[int, float], Union[int, float]]) -> None:
        """Add a value parameter.

        The default, min, and max value are all expected to be of the same value_type.
        The value_type is either an integer or floating-point. Conversions between the
        two during validation is available.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter, either int or float.
            default_value: the default value of the parameter.
            min_max_value: tuple with the minimum and maximum value of the parameter.
        """
        param = ConfigValueParam(
            name,
            value_type,
            default_value,
            min_max_value
        )

        self.add_param(param)
        self.values.append(param)

    def get_defaults(self) -> Dict[str, Any]:
        """Get the default values from all parameters.

        Returns:
            a dictionary containing name-default pairs for all parameters.
        """
        defaults = {}

        for param_name, param in self.params.items():
            defaults[param_name] = param.default_value

        return defaults

    def get_num_params(self) -> int:
        """Get the number of parameters.

        Returns:
            the parameter count.
        """
        return len(self.params)

    def get_param(self, param_name: str) -> Optional[ConfigParam]:
        """Get the parameter with the specified name.

        Returns:
             the parameter on success or None on failure.
        """
        return self.params.get(param_name)

    def get_param_names(self) -> List[str]:
        """Get the names of all parameters.

        Returns:
            a list of all parameter names.
        """
        param_names = []

        for param_name, _ in self.params.items():
            param_names.append(param_name)

        return param_names

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get a dictionary describing all the parameters.

        The parameters in the dictionary are stored separately:
            PARAM_KEY_OPTIONS: list of all option parameter descriptions.
            PARAM_KEY_VALUES: list of all value parameter descriptions.

        Returns:
            a dictionary containing the parameters' descriptions.
        """
        options = []
        for _, param in enumerate(self.options):
            options.append(param.to_dict())

        values = []
        for _, param in enumerate(self.values):
            values.append(param.to_dict())

        return {
            PARAM_KEY_OPTIONS: options,
            PARAM_KEY_VALUES: values
        }

    def add_param(self, param: ConfigParam) -> None:
        """Add a parameter to the internal dictionary.

        This function is only used internally by the parameters,
        use the other add functions instead.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            param: the parameter to add.
        """
        if param.name in self.params:
            raise KeyError('Config parameter already exists: ' + param.name)

        self.params[param.name] = param


def create_bool_param(name: str, default_value: bool):
    """Create a boolean option parameter.

    Args:
        name: the name of the boolean parameter.
        default_value: the default boolean value.

    Returns:
        the boolean parameter.
    """
    return ConfigOptionParam(name, bool, default_value, [True, False])


def create_empty_parameters() -> ConfigParameters:
    """Create the Config parameters with no entries."""
    return ConfigParameters()


def create_params_random_seed() -> ConfigParameters:
    """Create the configuration with only the random 'seed' parameter.

    Returns:
        the configuration parameters with one parameter: 'seed'.
    """
    params = ConfigParameters()
    params.add_random_seed('seed')
    return params
