"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import sys

PARAM_KEY_VALUES = 'values'
PARAM_KEY_OPTIONS = 'options'

PARAM_KEY_NAME = 'name'
PARAM_KEY_MIN = 'min'
PARAM_KEY_MAX = 'max'
PARAM_KEY_DEFAULT = 'default'


def create_value_param(name, min_val, max_val, default):
    """Creates a value parameter.

    Args:
        name(str): name of the parameter.
        min_val(int/float): minimum value of the parameter.
        max_val(int/float): maximum value of the parameter.
        default(int/float): default value of the parameter.

    Returns:
        (dict) with parameter name, min_val, max_val, default.
    """
    return {
        PARAM_KEY_NAME: name,
        PARAM_KEY_MIN: min_val,
        PARAM_KEY_MAX: max_val,
        PARAM_KEY_DEFAULT: default
    }


def create_option_param(name, options, default):
    """Creates an option parameter.

    Args:
        name(str): name of the parameter.
        options(array like): available options of the parameter.
        default(str): default option of the parameter.

    Returns:
        (dict) with parameter name, options, default.
    """
    return {
        PARAM_KEY_NAME: name,
        PARAM_KEY_OPTIONS: options,
        PARAM_KEY_DEFAULT: default
    }


def create_param_bool(name, default):
    """Creates a boolean option parameter.

    Args:
        name(str): name of the boolean parameter.
        default(bool): default value of the parameter.

    Returns:
        (dict) with parameter name, [True, False], default.
    """
    return create_option_param(name, [True, False], default)


def create_param_factors(factors):
    """Creates a factors value parameter.

    Args:
        factors(int): default value of the parameter.

    Returns:
        (dict) with parameter 'factors', 1, 100, factors.
    """
    return create_value_param('factors', 1, 100, factors)


def create_param_seed(name):
    """Creates a seed value parameter.

    Args:
        name(str): name of the parameter.

    Returns:
        (dict) with parameter name, 0, sys.maxsize, None)
    """
    return create_value_param(name, 0, sys.maxsize, None)


def get_param_defaults(params):
    """Gets the defaults of the specified parameters.

    Constructs a new dict with all the parameters set to their defaults.

    Args:
        params(dict): array like value and option parameters.

    Returns:
        (dict) parameter defaults for all values and options.
    """
    defaults = {}

    for _, value in enumerate(params[PARAM_KEY_VALUES]):
        defaults[value[PARAM_KEY_NAME]] = value[PARAM_KEY_DEFAULT]

    for _, option in enumerate(params[PARAM_KEY_OPTIONS]):
        defaults[option[PARAM_KEY_NAME]] = option[PARAM_KEY_DEFAULT]

    return defaults


def get_params_empty():
    """Gets the empty parameter dict.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [],
        PARAM_KEY_OPTIONS: []
    }
