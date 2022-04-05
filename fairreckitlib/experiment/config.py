"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.algorithms.common import FUNC_GET_ALGORITHM_PARAMS
from fairreckitlib.algorithms.elliot_alg.factory import get_elliot_recommender_factory
from fairreckitlib.algorithms.implicit_alg.factory import get_implicit_recommender_factory
from fairreckitlib.algorithms.lenskit_alg.factory import get_lenskit_recommender_factory
from fairreckitlib.algorithms.params import get_param_defaults
from .common import *


def create_config_dataset(dataset_name, test_ratio, split_type):
    return {
        EXP_KEY_DATASET_NAME: dataset_name,
        EXP_KEY_DATASET_PREFILTERS: [],
        EXP_KEY_DATASET_RATING_MODIFIER: None,
        EXP_KEY_DATASET_SPLIT: {
            EXP_KEY_DATASET_SPLIT_TEST_RATIO: test_ratio,
            EXP_KEY_DATASET_SPLIT_TYPE: split_type,
            EXP_KEY_DATASET_SPLIT_PARAMS: {}
        }
    }


def create_config_api_models(func_get_api_factory):
    api_name, api_factory = func_get_api_factory()

    models = []
    for model_name in api_factory:
        entry = api_factory[model_name]
        params = entry[FUNC_GET_ALGORITHM_PARAMS]()
        models.append({
            EXP_KEY_MODEL_NAME: model_name,
            EXP_KEY_MODEL_PARAMS: get_param_defaults(params)
        })

    return api_name, models


def create_config_all_models(elliot=True, implicit=True, lenskit=True):
    models = {}

    if elliot:
        elliot_api, elliot_models = create_config_api_models(get_elliot_recommender_factory)
        models[elliot_api] = elliot_models
    if implicit:
        implicit_api, implicit_models = create_config_api_models(get_implicit_recommender_factory)
        models[implicit_api] = implicit_models
    if lenskit:
        lenskit_api, lenskit_models = create_config_api_models(get_lenskit_recommender_factory)
        models[lenskit_api] = lenskit_models

    return models
