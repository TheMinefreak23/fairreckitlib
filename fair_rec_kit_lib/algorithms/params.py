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


def create_value_param(name, min, max, default):
    return {
        PARAM_KEY_NAME: name,
        PARAM_KEY_MIN: min,
        PARAM_KEY_MAX: max,
        PARAM_KEY_DEFAULT: default
    }


def create_option_param(name, options, default):
    return {
        PARAM_KEY_NAME: name,
        PARAM_KEY_OPTIONS: options,
        PARAM_KEY_DEFAULT: default
    }


def create_param_bool(name, default):
    return create_option_param(name, [True, False], default)


def create_param_factors(factors):
    return create_value_param('factors', 1, 100, factors)


def create_param_seed(name):
    return create_value_param(name, 0, sys.maxsize, None)


def get_param_defaults(params):
    defaults = {}

    for _, value in enumerate(params[PARAM_KEY_VALUES]):
        defaults[value[PARAM_KEY_NAME]] = value[PARAM_KEY_DEFAULT]

    for _, option in enumerate(params[PARAM_KEY_OPTIONS]):
        defaults[option[PARAM_KEY_NAME]] = option[PARAM_KEY_DEFAULT]

    return defaults
