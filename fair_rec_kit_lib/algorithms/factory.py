"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .elliot.factory import *
from .implicit.factory import *
from .lenskit.factory import *


def get_algorithm_list_from_factory(api_factory):
    algos = []

    for rec_name in api_factory:
        algos.append({
            ALGORITHM_NAME: rec_name,
            ALGORITHM_PARAMS: api_factory[rec_name][FUNC_GET_ALGORITHM_PARAMS]()
        })

    return algos


def get_recommender_api_list():
    rec_factory = get_recommender_factory()

    result = []

    for rec_api in rec_factory:
        api_factory = rec_factory[rec_api]

        result.append({
            'name': rec_api,
            'recommenders': get_algorithm_list_from_factory(api_factory)
        })

    return result


def get_recommender_factory():
    elliot_api, elliot_factory = get_elliot_recommender_factory()
    implicit_api, implicit_factory = get_implicit_recommender_factory()
    lenskit_api, lenskit_factory = get_lenskit_recommender_factory()

    return {
        elliot_api: elliot_factory,
        implicit_api: implicit_factory,
        lenskit_api: lenskit_factory,
    }
