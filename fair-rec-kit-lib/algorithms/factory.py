"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .elliot.factory import *
from .implicit.factory import *
from .lenskit.factory import *


def get_recommender_factory():
    elliot_api, elliot_factory = get_elliot_recommender_factory()
    implicit_api, implicit_factory = get_implicit_recommender_factory()
    lenskit_api, lenskit_factory = get_lenskit_recommender_factory()

    return {
        elliot_api: elliot_factory,
        implicit_api: implicit_factory,
        lenskit_api: lenskit_factory,
    }


def get_recommender_names():
    elliot_api, elliot_names = get_elliot_recommender_names()
    implicit_api, implicit_names = get_implicit_recommender_names()
    lenskit_api, lenskit_names = get_lenskit_recommender_names()

    return {
        elliot_api: elliot_names,
        implicit_api: implicit_names,
        lenskit_api: lenskit_names,
    }
