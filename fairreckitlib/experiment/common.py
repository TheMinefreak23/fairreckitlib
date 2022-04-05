"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.algorithms.common import ALGORITHM_NAME, ALGORITHM_PARAMS
from fairreckitlib.data.set import DATASET_LFM_360K, DATASET_LFM_1B, DATASET_LFM_2B
from fairreckitlib.data.set import DATASET_ML_100K, DATASET_ML_25M
from fairreckitlib.data.split.factory import SPLIT_RANDOM
from fairreckitlib.data.split.factory import SPLIT_TEMPORAL

EXP_KEY_NAME = 'name'

EXP_KEY_TYPE = 'type'
EXP_TYPE_RECOMMENDATION = 'recommendation'
EXP_TYPE_PREDICTION = 'prediction'

EXP_KEY_TOP_K = 'top_K'

EXP_KEY_DATASETS = 'datasets'
EXP_KEY_DATASET_NAME = 'name'
EXP_KEY_DATASET_PREFILTERS = 'prefilters'
EXP_KEY_DATASET_RATING_MODIFIER = 'rating_modifier'
EXP_KEY_DATASET_SPLIT = 'splitting'
EXP_KEY_DATASET_SPLIT_TEST_RATIO = 'test_ratio'
EXP_KEY_DATASET_SPLIT_TYPE = 'type'
EXP_KEY_DATASET_SPLIT_PARAMS = 'params'

EXP_KEY_MODELS = 'models'
EXP_KEY_MODEL_NAME = ALGORITHM_NAME
EXP_KEY_MODEL_PARAMS = ALGORITHM_PARAMS

EXP_KEY_EVALUATION = 'evaluation'
EXP_KEY_METRIC_NAME = 'name'
EXP_KEY_METRIC_PARAMS = 'params'
