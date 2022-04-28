"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.algorithms.constants import ALGORITHM_NAME
from fairreckitlib.algorithms.constants import ALGORITHM_PARAMS
from fairreckitlib.data.split.constants import SPLIT_PARAMS
from fairreckitlib.data.split.constants import SPLIT_TYPE
from fairreckitlib.data.split.factory import SPLIT_RANDOM

EXP_KEY_NAME = 'name'

EXP_KEY_TYPE = 'type'
EXP_TYPE_RECOMMENDATION = 'recommendation'
EXP_TYPE_PREDICTION = 'prediction'

EXP_KEY_TOP_K = 'top_K'
EXP_KEY_RATED_ITEMS_FILTER = 'rated_items_filter'

EXP_KEY_DATASETS = 'datasets'
EXP_KEY_DATASET_NAME = 'name'
EXP_KEY_DATASET_PREFILTERS = 'prefilters'
EXP_KEY_DATASET_RATING_MODIFIER = 'rating_modifier'
EXP_KEY_DATASET_SPLIT = 'splitting'
EXP_KEY_DATASET_SPLIT_PARAMS = SPLIT_PARAMS
EXP_KEY_DATASET_SPLIT_TEST_RATIO = 'test_ratio'
EXP_KEY_DATASET_SPLIT_TYPE = SPLIT_TYPE

EXP_KEY_MODELS = 'models'
EXP_KEY_MODEL_NAME = ALGORITHM_NAME
EXP_KEY_MODEL_PARAMS = ALGORITHM_PARAMS

EXP_KEY_EVALUATION = 'evaluation'
EXP_KEY_METRIC_NAME = 'name'
EXP_KEY_METRIC_PARAMS = 'params'
EXP_KEY_METRIC_PREFILTERS = 'prefilters'

EXP_KEY_METRIC_PARAM_K = 'K'

EXP_DEFAULT_SPLIT_TEST_RATIO = 0.2
EXP_DEFAULT_SPLIT_TYPE = SPLIT_RANDOM
EXP_DEFAULT_TOP_K = 10
EXP_DEFAULT_RATED_ITEMS_FILTER = True
