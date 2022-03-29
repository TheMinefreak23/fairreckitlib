"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def get_defaults_biased_mf():
    return {
        'features': 0,
        'iterations': 20,
        'reg': 0.1,
        'damping': 5.0,
        'bias': True,
        'method': 'cd',
        'random_seed': None
    }


def get_defaults_implicit_mf():
    return {
        'features': 3,
        'iterations': 20,
        'reg': 0.1,
        'weight': 40.0,
        'use_ratings': False,
        'method': 'cg',
        'random_seed': None
    }


def get_defaults_pop_score():
    return {
        'score_method': 'quantile'
    }


def get_defaults_random():
    return {
        'random_seed': None
    }
