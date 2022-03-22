""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def get_defaults_multi_vae():
    return {
        'epochs': 10,
        'batch_size': 512,
        'intermediate_dim': 600,
        'latent_dim': 200,
        'reg_lambda': 0.01,
        'lr': 0.001,
        'dropout_keep': 1
    }


def get_defaults_most_pop():
    return {}


def get_defaults_random():
    return {
        # 'random_seed': 42
    }


def get_defaults_svd_pure():
    return {
        'factors': 10
        # 'seed': 42
    }


def get_defaults_svd_pp():
    return {
        'epochs': 10,
        'batch_size': 512,
        'factors': 50,
        'lr': 0.001,
        'reg_w': 0.1,
        'reg_b': 0.001
    }


def get_defaults_item_knn():
    return {
        'neighbours': 40,
        'similarity': 'cosine',
        'implementation': 'aiolli'
    }


def get_defaults_user_knn():
    return {
        'neighbours': 40,
        'similarity': 'cosine',
        'implementation': 'aiolli'
    }
