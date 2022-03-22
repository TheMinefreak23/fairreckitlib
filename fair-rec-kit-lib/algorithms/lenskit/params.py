""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def get_defaults_als_biased_mf():
    return {
        'features': 0,
        'iterations': 20,
        'reg': 0.1,
        'damping': 5.0,
        'bias': True,
        'method': 'cd'
    }


def get_defaults_als_implicit_mf():
    return {
        'features': 3,
        'iterations': 20,
        'reg': 0.1,
        'weight': 40.0,
        'use_ratings': False,
        'method': 'cg'
    }


def get_defaults_pop_score():
    return { 'score_method': 'quantile' }


def get_defaults_bias():
    return {
        'items': True,
        'users': True,
        'damping': 0.0,
    }


def get_defaults_funk_svd():
    return {
        'features': 0,
        'iterations': 100,
        'lrate': 0.001,
        'reg': 0.015,
        'damping': 5.0,
        'range': None,
        'bias': True
    }


def get_defaults_knn_item_item():
    return {
        'max_nnbrs': 20,
        'min_nbrs': 1,
        'min_sim': 1e-06,
        'save_nbrs': None,
        'feedback': 'implicit',
        # center=bool # explicit => True / implicit => False
        # aggregate=str # explicit => weighted-average / implicit => sum
        # use_ratings=bool # explicit => True / implicit => False
    }


def get_defaults_knn_user_user():
    return {
        'max_nnbrs': 20,
        'min_nbrs': 1,
        'min_sim': 0.0,
        'feedback': 'implicit',
        # center=bool # explicit => True / implicit => False
        # aggregate=str # explicit => weighted-average / implicit => sum
        # use_ratings=bool # explicit => True / implicit => False
    }
