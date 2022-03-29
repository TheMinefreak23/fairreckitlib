"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def get_defaults_alternating_least_squares():
    return {
        'factors': 100,
        'regularization': 0.01,
        'use_native': True,
        'use_cg': True,
        'iterations': 15,
        'calculate_training_loss': False,
        'random_seed': None
    }


def get_defaults_bayesian_personalized_ranking():
    return {
        'factors': 100,
        'learning_rate': 0.01,
        'regularization': 0.01,
        'iterations': 100,
        'verify_negative_samples': True,
        'random_seed': None
    }


def get_defaults_logistic_matrix_factorization():
    return {
        'factors': 30,
        'learning_rate': 1.00,
        'regularization': 0.6,
        'iterations': 30,
        'neg_prop': 30,
        'random_seed': None
    }
