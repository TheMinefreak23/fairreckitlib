"""This module contains a container of configuration parameters.

Classes:

    ConfigParameters: container that stores multiple ConfigParam's.

Functions:

    create_empty_parameters: create parameters with no entries.
    create_params_random_seed: create configuration with only a random seed parameter.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..core_constants import KEY_RANDOM_SEED
from .config_base_param import ConfigParam, PARAM_KEY_OPTIONS, PARAM_KEY_VALUES
from .config_option_param import ConfigSingleOptionParam, ConfigMultiOptionParam, create_bool_param
from .config_value_param import ConfigNumberParam, ConfigRandomParam, ConfigRangeParam


class ConfigParameters:
    """Config Parameters.

    Container with varying Config parameters using a dictionary.
    Moreover, he added option/value parameters are stored separately.

    Public methods:

    add_bool
    add_multi_option
    add_single_option
    add_number
    add_random_seed
    add_range
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

    def add_multi_option(
            self,
            name: str,
            default_value: List[str],
            options: List[str]) -> None:
        """Add a multi option parameter.

        The default_value and all the options are expected to be strings or None.
        The default_value list is expected to have at least one entry.
        The default_value list is expected to be present in the list of available options.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the parameter.
            default_value: the default option(s) of the parameter.
            options: list of available options for the parameter.
        """
        param = ConfigMultiOptionParam(
            name,
            default_value,
            options
        )

        self.add_param(param)
        self.options.append(param)

    def add_single_option(
            self,
            name: str,
            value_type: Type,
            default_option: str,
            options: List[str]) -> None:
        """Add a single option parameter.

        The default_value and all the options are expected to be strings.
        The default_value is expected to be present in the list of available options.
        The options list is expected to have unique values.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the parameter.
            value_type:  the type of the parameter.
            default_option: default option of the parameter.
            options: list of available options for the parameter.
        """
        param = ConfigSingleOptionParam(
            name,
            value_type,
            default_option,
            options
        )

        self.add_param(param)
        self.options.append(param)

    def add_number(
            self,
            name: str,
            value_type: Type,
            default_value: Union[int, float],
            min_max_value: Union[Tuple[int, int], Tuple[float, float]]) -> None:
        """Add a number parameter.

        The value_type of default_value and min_max_value types are all expected to be either
        int or float, conversions between the two during validation is available.
        The default value is expected to be between the min_max_value.
        The min_max_value is expected to have min_value <= max_value.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: the default value of the parameter.
            min_max_value: tuple with the minimum and maximum value of the parameter.
        """
        param = ConfigNumberParam(
            name,
            value_type,
            default_value,
            min_max_value
        )

        self.add_param(param)
        self.values.append(param)

    def add_random_seed(self, name: str) -> None:
        """Add a random seed parameter.

        An integer number parameter, where the default_value is None.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the random seed parameter.
        """
        param = ConfigRandomParam(name)

        self.add_param(param)
        self.values.append(param)

    def add_range(
            self,
            name: str,
            value_type: Type,
            default_value: Union[Tuple[int, int], Tuple[float, float]],
            min_max_value: Union[Tuple[int, int], Tuple[float, float]]) -> None:
        """Add a range parameter.

        The value_type of default_value and min_max_value types are all expected to be either
        int or float, conversions between the two during validation is available.
        The default_value min and max values are expected to be between the min_max_value.
        The default_value and min_max_value are expected to have min_value <= max_value.
        Raises a KeyError when the name of the parameter is already present.

        Args:
            name: the name of the parameter.
            value_type: the type of the parameter.
            default_value: tuple with the default minimum and maximum value of the parameter.
            min_max_value: tuple with the minimum and maximum value of the parameter.
        """
        param = ConfigRangeParam(
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
        for param in self.options:
            options.append(param.to_dict())

        values = []
        for param in self.values:
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


def create_empty_parameters() -> ConfigParameters:
    """Create the Config parameters with no entries."""
    return ConfigParameters()


def create_params_random_seed() -> ConfigParameters:
    """Create the configuration with only the random 'seed' parameter.

    Returns:
        the configuration parameters with one parameter: 'seed'.
    """
    params = ConfigParameters()
    params.add_random_seed(KEY_RANDOM_SEED)
    return params
