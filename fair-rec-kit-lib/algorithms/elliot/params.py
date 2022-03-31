"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..params import *


def get_params_funk_svd():
    return {
        PARAM_KEY_VALUES: [
            _create_param_epochs(),
            _create_param_batch_size(),
            create_param_factors(10),
            _create_param_learning_rate(),
            _create_param_reg_w(),
            _create_param_reg_b(),
            create_param_seed('seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_params_item_knn():
    return {
        PARAM_KEY_VALUES: [
            _create_param_neighbours()
        ],
        PARAM_KEY_OPTIONS: [
            _create_param_similarity(),
            _create_param_implementation()
        ]
    }


def get_params_multi_vae():
    return {
        PARAM_KEY_VALUES: [
            _create_param_batch_size(),
            _create_param_epochs(),
            _create_param_learning_rate(),
            _create_param_reg_lambda(),
            create_value_param('intermediate_dim', 100, 1000, 600),
            create_value_param('latent_dim', 100, 500, 200),
            create_value_param('dropout_keep', 0.0, 1.0, 1.0),
            create_param_seed('seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_params_most_pop():
    return {
        PARAM_KEY_VALUES: [],
        PARAM_KEY_OPTIONS: []
    }


def get_params_pure_svd():
    return {
        PARAM_KEY_VALUES: [
            create_param_factors(10),
            create_param_seed('seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_params_random():
    return {
        PARAM_KEY_VALUES: [
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_params_svd_pp():
    return {
        PARAM_KEY_VALUES: [
            _create_param_epochs(),
            _create_param_batch_size(),
            create_param_factors(50),
            _create_param_learning_rate(),
            _create_param_reg_w(),
            _create_param_reg_b(),
            create_param_seed('seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def get_params_user_knn():
    return {
        PARAM_KEY_VALUES: [
            _create_param_neighbours()
        ],
        PARAM_KEY_OPTIONS: [
            _create_param_similarity(),
            _create_param_implementation()
        ]
    }


def _create_param_batch_size():
    return create_value_param('batch_size', 100, 10000, 512)


def _create_param_epochs():
    return create_value_param('epochs', 1, 50, 10)


def _create_param_implementation():
    options = [
        'aiolli',
        'classical'
    ]

    return create_option_param('implementation', options, options[0])


def _create_param_learning_rate():
    return create_value_param('lr', 0.0001, 1.0, 0.001)


def _create_param_neighbours():
    return create_value_param('neighbours', 1, 50, 40)


def _create_param_reg_b():
    return create_value_param('reg_b', 0.0001, 1.0, 0.001)


def _create_param_reg_w():
    return create_value_param('reg_w', 0.0001, 1.0, 0.1)


def _create_param_reg_lambda():
    return create_value_param('reg_lambda', 0.0001, 1.0, 0.01)


def _create_param_similarity():
    options = [
        'cosine',
        'adjusted',
        'asymmetric',
        'pearson',
        'jaccard',
        'dice',
        'tversky',
        'tanimoto'
    ]

    return create_option_param('similarity', options, options[0])
