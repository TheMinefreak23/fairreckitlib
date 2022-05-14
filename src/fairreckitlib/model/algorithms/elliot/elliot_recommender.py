"""
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

    def train(self, train_set: pd.DataFrame) -> None:
        """Train the elliot model not supported."""
        # not used, training is done by running the framework
        raise NotImplementedError()

    def recommend(self, user: int, num_items: int=10) -> pd.DataFrame:
        """Recommend with the elliot model not supported."""
        # not used, recommending is done by running the framework
        raise NotImplementedError()

    def recommend_batch(self, users: List[int], num_items: int=10) -> pd.DataFrame:
        """Recommend batching with the elliot model not supported."""
        # not used, recommending is done by running the framework
        raise NotImplementedError()


def create_funk_svd(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the FunkSVD recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): number of iterations.
            batch_size(int): batch size of the computation.
            factors(int): number of factors of feature embeddings.
            learning_rate(float): the learning rate.
            reg_w(float): regularization coefficient for latent factors.
            reg_b(flot): regularization coefficient for bias.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of FunkSVD.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ElliotRecommender(name, params, **kwargs)


def create_item_knn(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the ItemKNN recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            neighbours(int): number of item neighbors.
            similarity(str): similarity function to use.
            implementation(str): implementation type (‘aiolli’ or ‘classical’).

    Returns:
        the ElliotRecommender wrapper of ItemKNN.
    """
    return ElliotRecommender(name, params, **kwargs)


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
        name(str): the name of the algorithm.
        params(dict): with the entries:
            epochs(int): number of iterations.
            batch_size(int): batch size of the computation.
            learning_rate(float): the learning rate.
            reg_lambda(float): regularization coefficient.
            intermediate_dim(int): number of intermediate dimension.
            latent_dim(int): number of latent factors.
            dropout_pkeep(float): the dropout probability.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of MultiVAE.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ElliotRecommender(name, params, **kwargs)


def create_pure_svd(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the PureSVD recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors(int): number of latent factors.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of PureSVD.
    """
    return ElliotRecommender(name, params, **kwargs)


def create_random(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the Random recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of Random.
    """
    if params['seed'] is None:
        params['random_seed'] = int(time.time())
    else:
        params['random_seed'] = params['seed']
        del params['seed']

    return ElliotRecommender(name, params, **kwargs)


def create_svd_pp(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the SVDpp recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): number of iterations.
            batch_size(int): batch size of the computation.
            factors(int): number of latent factors.
            learning_rate(float): the learning rate.
            reg_w(float): regularization coefficient for latent factors.
            reg_b(float): regularization coefficient for bias.
            seed(int): the random seed or None for the current time as seed.

    Returns:
        the ElliotRecommender wrapper of SVDpp.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ElliotRecommender(name, params, **kwargs)


def create_user_knn(name: str, params: Dict[str, Any], **kwargs) -> ElliotRecommender:
    """Create the UserKNN recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            neighbours(int): number of user neighbors.
            similarity(str): similarity function to use.
            implementation(str): implementation type (‘aiolli’ or ‘classical’).

    Returns:
        the ElliotRecommender wrapper of UserKNN.
    """
    return ElliotRecommender(name, params, **kwargs)
