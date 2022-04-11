"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..params import PARAM_KEY_OPTIONS
from ..params import PARAM_KEY_VALUES
from ..params import create_option_param
from ..params import create_param_bool
from ..params import create_param_seed
from ..params import create_value_param


def get_lenskit_params_biased_mf():
    """Gets the parameters of the BiasedMF algorithm.

    Returns:
        (dict) containing array like values and options.
    """
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


def get_lenskit_params_implicit_mf():
    """Gets the parameters of the ImplicitMF algorithm.

    Returns:
        (dict) containing array like values and options.
    """
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


def get_lenskit_params_item_item():
    """Gets the parameters of the ItemItem algorithm.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [
            _create_param_max_nnbrs(),
            _create_param_min_nnbrs(),
            _create_param_min_sim(1e-06)
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_lenskit_params_pop_score():
    """Gets the parameters of the PopScore algorithm.

    Returns:
        (dict) containing array like values and options.
    """
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


def get_lenskit_params_random():
    """Gets the parameters of the Random algorithm.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_lenskit_params_user_user():
    """Gets the parameters of the UserUser algorithm.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [
            _create_param_max_nnbrs(),
            _create_param_min_nnbrs(),
            _create_param_min_sim(0.0)
        ],
        PARAM_KEY_OPTIONS: []
    }


def _create_param_features(min_features):
    return create_value_param('features', min_features, 50, 10)


def _create_param_iterations():
    return create_value_param('iterations', 1, 50, 20)


def _create_param_max_nnbrs():
    return create_value_param('max_nnbrs', 1, 50, 10)


def _create_param_min_nnbrs():
    return create_value_param('min_nbrs', 1, 50, 1)


def _create_param_min_sim(min_sim):
    return create_value_param('min_sim', 0.0, 10.0, min_sim)


def _create_param_reg(name):
    return create_value_param(name, 0.0001, 1.0, 0.1)
