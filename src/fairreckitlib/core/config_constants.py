"""This module contains configuration constants that are used in other packages.

Constants:

    KEY_NAME: the key that is used for a name.
    KEY_PARAMS: the key that is used for params.
    KEY_TYPE: the key that is used for types.

    TYPE_PREDICTION: the prediction experiment type.
    TYPE_RECOMMENDATION: the recommender experiment type.

    VALID_TYPES: the valid experiment types.

    KEY_TOP_K: the key that is used for top k.
    KEY_RATED_ITEMS_FILTER: the key that is used for the rated items filter.

    DEFAULT_TOP_K: the default top k for recommender experiments.
    DEFAULT_RATED_ITEMS_FILTER: the default rated items filter for recommender experiments.

    MODEL_USER_BATCH_SIZE: the batch size of users that is used when model computations are done.
    MODEL_RATINGS_FILE: the file that is used to store the computed model ratings.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

KEY_NAME = 'name'
KEY_PARAMS = 'params'
KEY_TYPE = 'type'

TYPE_PREDICTION = 'prediction'
TYPE_RECOMMENDATION = 'recommendation'

VALID_TYPES = [TYPE_PREDICTION, TYPE_RECOMMENDATION]

KEY_TOP_K = 'top_K'
KEY_RATED_ITEMS_FILTER = 'rated_items_filter'

DEFAULT_TOP_K = 10
DEFAULT_RATED_ITEMS_FILTER = True

MODEL_USER_BATCH_SIZE = 10000
MODEL_RATINGS_FILE = 'ratings.tsv'
