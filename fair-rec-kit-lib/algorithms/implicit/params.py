""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


def get_defaults_als():
    return {
        'factors': 100,
        'regularization': 0.01,
        'use_native': True,
        'use_cg': True,
        'iterations': 15,
        'calculate_training_loss': False
    }


def get_defaults_bpr():
    return {
        'factors': 100,
        'learning_rate': 0.01,
        'regularization': 0.01,
        'iterations': 100,
        'verify_negative_samples': True
    }


def get_defaults_lmf():
    return {
        'factors': 30,
        'learning_rate': 1.00,
        'regularization': 0.6,
        'iterations': 30,
        'neg_prop': 30
    }


def get_defaults_cosine():
    return {
        'K': 20
    }


def get_defaults_tfidf():
    return {
        'K': 20
    }


def get_defaults_bm25():
    return {
        'K': 20,
        'K1': 1.2,
        'B': 0.75
    }
