"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..params import *


def get_params_biased_mf():
    methods = [
        'cd',
        'lu'
    ]

    return {
        PARAM_KEY_VALUES: [
            _create_param_features(0),
            _create_param_iterations(),
            _create_param_reg('user_reg'),
            _create_param_reg('item_reg'),
            create_value_param('damping', 0.0, 1000.0, 5.0),
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: [
            create_option_param('method', methods, methods[0])
        ]
    }


def get_params_implicit_mf():
    methods = [
        'cg',
        'lu'
    ]

    return {
        PARAM_KEY_VALUES: [
            _create_param_features(3),
            _create_param_iterations(),
            _create_param_reg('reg'),
            create_value_param('weight', 1.0, 10000.0, 40.0),
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: [
            create_option_param('method', methods, methods[0]),
            create_param_bool('use_ratings', False)
        ]
    }


def get_params_pop_score():
    options = [
        'quantile',
        'rank',
        'count'
    ]

    return {
        PARAM_KEY_VALUES: [],
        PARAM_KEY_OPTIONS: [
            create_option_param('score_method', options, options[0])
        ]
    }


def get_params_random():
    return {
        PARAM_KEY_VALUES: [
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def _create_param_features(min):
    return create_value_param('features', min, 50, 10)


def _create_param_iterations():
    return create_value_param('iterations', 1, 50, 20)


def _create_param_reg(name):
    return create_value_param(name, 0.0001, 1.0, 0.1)
