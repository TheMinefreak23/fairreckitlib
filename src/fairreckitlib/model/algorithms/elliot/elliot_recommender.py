"""This module contains the elliot recommender and creation functions.

Classes:

    ElliotRecommender: recommender implementation for elliot.

Functions:

    create_funk_svd: create FunkSVD recommender (factory creation compatible).
    create_item_knn: create ItemKNN recommender (factory creation compatible).
    create_most_pop: create MostPop recommender (factory creation compatible).
    create_multi_vae: create MultiVAE recommender (factory creation compatible).
    create_pure_svd: create PureSVD recommender (factory creation compatible).
    create_random: create Random recommender (factory creation compatible).
    create_user_knn: create UserKNN recommender (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Dict, List

import pandas as pd

from ..base_recommender import Recommender


class ElliotRecommender(Recommender):
    """Recommender implementation for the Elliot framework."""

    def __init__(self, name: str, params: Dict[str, Any], **kwargs):
        """Construct the elliot recommender.

        The recommender is not procedural, instead it serves as a wrapper
        that holds the correct parameters used by the framework which are
        used in the model pipeline.

        Args:
            name: the name of the recommender.
            params: the parameters of the recommender.

        Keyword Args:
            num_threads(int): the max number of threads the recommender can use.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.
        """
        Recommender.__init__(self, name, params, kwargs['num_threads'],
                             kwargs['rated_items_filter'])
        if not self.rated_items_filter:
            raise RuntimeError('no rated items filter is not supported.')

    def on_train(self) -> None:
        """Train the elliot model not supported."""
        raise RuntimeError('training is done by running the framework')

    def on_recommend(self, user: int, num_items: int) -> pd.DataFrame:
        """Recommend with the elliot model not supported."""
        raise RuntimeError('recommending is done by running the framework')

    def on_recommend_batch(self, users: List[int], num_items: int) -> pd.DataFrame:
        """Recommend batching with the elliot model not supported."""
        raise RuntimeError('recommending is done by running the framework')


def create_funk_svd(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the FunkSVD recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): number of iterations.
            factors(int): number of factors of feature embeddings.
            learning_rate(float): the learning rate.
            regularization_factors(float): regularization coefficient for latent factors.
            regularization_bias(flot): regularization coefficient for bias.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of FunkSVD.
    """
    elliot_params = {
        'epochs': params['iterations'],
        'batch_size': 512,
        'factors': params['factors'],
        'lr': params['learning_rate'],
        'reg_w': params['regularization_factors'],
        'reg_b': params['regularization_bias'],
        'seed': params['seed'] if params.get('seed') else int(time.time())
    }

    return ElliotRecommender(name, elliot_params, **kwargs)


def create_item_knn(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the ItemKNN recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            neighbors(int): number of item neighbors.
            similarity(str): similarity function to use.
            implementation(str): implementation type (‘aiolli’ or ‘classical’).

    Returns:
        the ElliotRecommender wrapper of ItemKNN.
    """
    elliot_params = {
        'neighbors': params['neighbors'],
        'similarity': params['similarity'],
        'implementation': params['implementation'],
    }

    return ElliotRecommender(name, elliot_params, **kwargs)


def create_most_pop(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the MostPop recommender.

    Args:
        name: the name of the algorithm.
        params: there are no parameters for this algorithm.

    Returns:
        the ElliotRecommender wrapper of MostPop.
    """
    return ElliotRecommender(name, params, **kwargs)


def create_multi_vae(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the MultiVAE recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            iterations(int): number of iterations.
            factors(int): number of latent factors.
            learning_rate(float): the learning rate.
            intermediate_dimensions(int): number of intermediate dimension.
            regularization_factors(float): regularization coefficient.
            dropout_probability(float): the dropout probability.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of MultiVAE.
    """
    elliot_params = {
        'epochs': params['iterations'],
        'batch_size': 512,
        'lr': params['learning_rate'],
        'reg_lambda': params['regularization_factors'],
        'intermediate_dim': params['intermediate_dimensions'],
        'latent_dim': params['factors'],
        'dropout_pkeep': params['dropout_probability'],
        'seed': params['seed'] if params.get('seed') else int(time.time())
    }

    return ElliotRecommender(name, elliot_params, **kwargs)


def create_pure_svd(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the PureSVD recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): number of latent factors.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of PureSVD.
    """
    elliot_params = {
        'factors': params['factors'],
        'seed': params['seed'] if params.get('seed') else int(time.time())
    }

    return ElliotRecommender(name, elliot_params, **kwargs)


def create_random(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the Random recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of Random.
    """
    elliot_params = {
        'random_seed': params['seed'] if params.get('seed') else int(time.time())
    }

    return ElliotRecommender(name, elliot_params, **kwargs)


def create_svd_pp(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the SVDpp recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            iterations(int): number of iterations.
            factors(int): number of latent factors.
            learning_rate(float): the learning rate.
            regularization_factors(float): regularization coefficient for latent factors.
            regularization_bias(float): regularization coefficient for bias.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of SVDpp.
    """
    elliot_params = {
        'epochs': params['iterations'],
        'batch_size': 512,
        'factors': params['factors'],
        'lr': params['learning_rate'],
        'reg_w': params['regularization_factors'],
        'reg_b': params['regularization_bias'],
        'seed': params['seed'] if params.get('seed') else int(time.time())
    }

    return ElliotRecommender(name, elliot_params, **kwargs)


def create_user_knn(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the UserKNN recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            neighbors(int): number of user neighbors.
            similarity(str): similarity function to use.
            implementation(str): implementation type (‘aiolli’ or ‘classical’).

    Returns:
        the ElliotRecommender wrapper of UserKNN.
    """
    elliot_params = {
        'neighbors': params['neighbors'],
        'similarity': params['similarity'],
        'implementation': params['implementation'],
    }

    return ElliotRecommender(name, elliot_params, **kwargs)
