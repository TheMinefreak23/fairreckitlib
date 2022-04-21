"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import sys

PARAM_KEY = 'key'
PARAM_KEY_NAME = 'name'
PARAM_KEY_DEFAULT = 'default'
PARAM_KEY_MIN = 'min'
PARAM_KEY_MAX = 'max'
PARAM_KEY_OPTIONS = 'options'
PARAM_KEY_VALUES = 'values'


class AlgorithParam(metaclass=ABCMeta):
    """Algorithm Param base class.

    Args:
        name(str): name of the parameter.
        value_type(type): type of the parameter.
        default_value(object): default option of the parameter.
    """
    def __init__(self, name, value_type, default_value):
        self.name = name
        self.value_type = value_type
        self.default_value = default_value

    @abstractmethod
    def to_dict(self):
        """Gets a dictionary describing the parameter.

        Returns:
            (dict): containing the parameter descriptions.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate_value(self, value):
        """Validates the specified value with the parameter.

        Args:
            value(object): the value to verify with the parameter.

        Returns:
            (bool): whether the validation was successful.
            (object): the validated value when not correct.
            (str): message containing extra information about the validation.
        """
        raise NotImplementedError()


class AlgorithmOptionParam(AlgorithParam):
    """Algorithm Option Parameter.

    The default value and all the options are expected
    to be of the same value_type.

    Args:
        name(str): name of the parameter.
        value_type(type): type of the parameter.
        default_value(object): default option of the parameter.
        options(array like): list of available options for the parameter.
    """
    def __init__(self, name, value_type, default_value, options):
        AlgorithParam.__init__(self, name, value_type, default_value)
        self.options = options

    def to_dict(self):
        return {
            PARAM_KEY_NAME: self.name,
            PARAM_KEY_OPTIONS: self.options,
            PARAM_KEY_DEFAULT: self.default_value
        }

    def validate_value(self, value):
        if value is None:
            return False, self.default_value, \
                   'no value specified, expected one of: ' + str(self.options)

        # type check
        if type(value) != self.value_type:
            return False, self.default_value, \
                   'expected ' + str(self.value_type) + ' got '+ str(type(value))

        # value check
        if value in self.options:
            return True, value, ''

        return False, self.default_value, 'expected one of: ' + str(self.options)


class AlgorithmValueParam(AlgorithParam):
    """Algorithm Value Parameter.

    The default, min, and max value are all expected to be of the same value_type.
    The value_type is either an integer or floating-point. Conversions between the
    two during validation is available.

    Args:
        name(str): name of the parameter.
        value_type(int/float): type of the parameter.
        default_value(int/float): default value of the parameter.
        min_value(int/float): minimum value of the parameter.
        max_value(int/float): maximum value of the parameter.
    """
    def __init__(self, name, value_type, default_value, min_value, max_value):
        AlgorithParam.__init__(self, name, value_type, default_value)
        self.min_value = min_value
        self.max_value = max_value

    def to_dict(self):
        return {
            PARAM_KEY_NAME: self.name,
            PARAM_KEY_MIN: self.min_value,
            PARAM_KEY_MAX: self.max_value,
            PARAM_KEY_DEFAULT: self.default_value
        }

    def validate_value(self, value):
        if value is None:
            return False, self.default_value, \
                   'no value specified, expected value between ' + \
                   str(self.min_value) + ' and ' + str(self.max_value)

        error = ''
        # type checks
        if isinstance(value, float) and self.value_type == int:
            value = self.value_type(value)
            error = '(value cast to int) '
        elif isinstance(value, int) and self.value_type == float:
            value = self.value_type(value)
            error = '(value cast to float) '
        elif type(value) != self.value_type:
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


class AlgorithmRandomParam(AlgorithmValueParam):
    """Algorithm Random Parameter.

    Derived from an integer AlgorithmValueParam, and in addition
    allows the default_value to be None.

    Args:
        name(str): name of the random seed parameter.
    """
    def __init__(self, name):
        AlgorithmValueParam.__init__(self, name, int, None, 0, sys.maxsize)

    def validate_value(self, value):
        # skips the 'None' error from AlgorithmValueParam
        if value is None:
            return True, value, ''

        return AlgorithmValueParam.validate_value(self, value)


class AlgorithmParameters:
    """Algorithm Parameters.

    Container with varying algorithm parameters using a dictionary.
    Moreover, he added option/value parameters are stored separately.
    """
    def __init__(self):
        self.params = {}
        self.options = []
        self.values = []

    def add_bool(self, name, default_value):
        """Adds a boolean parameter.

        Sugar for an option parameter that is either True or False.

        Raises a KeyError when the name of the parameter is already present.

        Args:
              name(str): name of the boolean parameter.
              default_value(bool): the default boolean value.
        """
        self.add_option(name, bool, default_value, [True, False])

    def add_option(self, name, value_type, default_option, options):
        """Adds an option parameter.

        The default value and all the options are expected
        to be of the same value_type.

        Raises a KeyError when the name of the parameter is already present.

        Args:
            name(str): name of the parameter.
            value_type(type): type of the parameter.
            default_option(object): default option of the parameter.
            options(array like): list of available options for the parameter.
        """
        param = AlgorithmOptionParam(
            name,
            value_type,
            default_option,
            options
        )

        self.__add_param(param)
        self.options.append(param)

    def add_random_seed(self, name):
        """Adds a random seed parameter.

        An integer value parameter, where the default_value is None.

        Raises a KeyError when the name of the parameter is already present.

        Args:
            name(str): name of the random seed parameter.
        """
        param = AlgorithmRandomParam(name)

        self.__add_param(param)
        self.values.append(param)

    def add_value(self, name, value_type, default_value, min_value, max_value):
        """Adds a value parameter.

        The default, min, and max value are all expected to be of the same value_type.
        The value_type is either an integer or floating-point. Conversions between the
        two during validation is available.

        Raises a KeyError when the name of the parameter is already present.

        Args:
            name(str): name of the parameter.
            value_type(int/float): type of the parameter.
            default_value(int/float): default value of the parameter.
            min_value(int/float): minimum value of the parameter.
            max_value(int/float): maximum value of the parameter.
        """
        param = AlgorithmValueParam(
            name,
            value_type,
            default_value,
            min_value,
            max_value
        )

        self.__add_param(param)
        self.values.append(param)

    def get_defaults(self):
        """Gets the default parameters.

        Returns:
            defaults(dict): containing name-default pairs for all parameters.
        """
        defaults = {}

        for param_name, param in self.params.items():
            defaults[param_name] = param.default_value

        return defaults

    def get_param(self, param_name):
        """Gets the parameter with the specified name.

        Returns:
            param(AlgorithmParam): the parameter on success or None on failure.
        """
        return self.params.get(param_name)

    def get_param_names(self):
        """Gets the names of all parameters.

        Returns:
            param_names(array like): list of all parameter names.
        """
        param_names = []

        for param_name, _ in self.params.items():
            param_names.append(param_name)

        return param_names

    def to_dict(self):
        """Gets a dictionary describing all parameters.

        The parameters in the dictionary are stored separately:
            PARAM_KEY_OPTIONS: list of all option parameter descriptions.
            PARAM_KEY_VALUES: list of all value parameter descriptions.

        Returns:
            (dict): containing the parameters' descriptions.
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

    def __add_param(self, param):
        """Adds a parameter to the internal dictionary.

        Raises a KeyError when the name of the parameter is already present.

        Args:
            param(AlgorithParam): the parameter to add.
        """
        if param.name in self.params:
            raise KeyError('Algorithm parameter already exists: ', + param.name)

        self.params[param.name] = param


def get_empty_parameters():
    """Gets the algorithm parameters with no entries."""
    return AlgorithmParameters()
