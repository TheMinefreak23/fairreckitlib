"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

import lenskit
from lenskit import batch
import pandas as pd

from ..base_predictor import Predictor
from . import lenskit_algorithms


class LensKitPredictor(Predictor):
    """Predictor implementation for the LensKit framework."""

    def __init__(self, algo: lenskit.Predictor, name: str, params: Dict[str, Any], **kwargs):
        """Construct the lenskit predictor.

        Args:
            algo: the lenskit prediction algorithm.
            name: the name of the predictor.
            params: the parameters of the predictor.

        Keyword Args:
            num_threads(int): the max number of threads the predictor can use.
        """
        Predictor.__init__(self, name, params, kwargs['num_threads'])
        self.algo = algo

    def train(self, train_set: pd.DataFrame) -> None:
        """Fit the lenskit algorithm on the specified train set.

        Args:
            train_set: with at least three columns: 'user', 'item', 'rating'.
        """
        self.algo.fit(train_set)

    def predict(self, user: int, item: int) -> float:
        """Compute a prediction for the specified user and item.

        Lenskit predictors allow for predicting multiple items at the same time.
        To conform with the interface only one item needs to be predicted and all
        the extra data that it generates needs to be excluded.

        Args:
            user: the user ID.
            item: the item ID.

        Returns:
            the prediction rating.
        """
        prediction = self.algo.predict_for_user(user, [item])
        return prediction[item]

    def predict_batch(self, user_item_pairs: pd.DataFrame) -> pd.DataFrame:
        """Compute the predictions for each of the specified user and item pairs.

        Lenskit predictors have a batch implementation available that allows for
        predicting ratings using multiple 'jobs'.

        Args:
            user_item_pairs: with at least two columns: 'user', 'item'.

        Returns:
            dataFrame with the columns: 'user', 'item', 'prediction'.
        """
        n_jobs = self.num_threads if self.num_threads > 0 else None
        predictions = batch.predict(self.algo, user_item_pairs, n_jobs=n_jobs)
        return predictions[['user', 'item', 'prediction']]


def create_biased_mf(name: str, params: Dict[str, Any], **kwargs) -> LensKitPredictor:
    """Create the BiasedMF predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            features(int): the number of features to train.
            iterations(int): the number of iterations to train.
            user_reg(float): the regularization factor for users.
            item_reg(float): the regularization factor for items.
            damping(float): damping factor for the underlying bias.
            method(str): the solver to use ('cd' or 'lu').
            random_seed(int): the random seed or None for the current time as seed.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the LensKitPredictor wrapper of BiasedMF.
    """
    algo = lenskit_algorithms.create_biased_mf(params)
    return LensKitPredictor(algo, name, params, **kwargs)


def create_implicit_mf(name: str, params: Dict[str, Any], **kwargs) -> LensKitPredictor:
    """Create the ImplicitMF predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            features(int): the number of features to train.
            iterations(int): the number of iterations to train.
            reg(float): the regularization factor.
            weight(flot): the scaling weight for positive samples.
            use_ratings(bool): whether to use the rating column or treat
                every rated user-item pair as having a rating of 1.
            method(str): the training method ('cg' or 'lu').
            random_seed(int): the random seed or None for the current time as seed.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the LensKitPredictor wrapper of ImplicitMF.
    """
    algo = lenskit_algorithms.create_implicit_mf(params)
    return LensKitPredictor(algo, name, params, **kwargs)


def create_item_item(name: str, params: Dict[str, Any], **kwargs) -> LensKitPredictor:
    """Create the ItemItem predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_neighbors(int): the maximum number of neighbors for scoring each item.
            min_neighbors(int): the minimum number of neighbors for scoring each item.
            min_similarity(float): minimum similarity threshold for considering a neighbor.
            feedback(str): control how feedback should be interpreted ('explicit' or 'implicit').

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the LensKitPredictor wrapper of ItemItem.
    """
    algo = lenskit_algorithms.create_item_item(params)
    return LensKitPredictor(algo, name, params, **kwargs)


def create_pop_score(name: str, params: Dict[str, Any], **kwargs) -> LensKitPredictor:
    """Create the PopScore predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            score_method(str): for computing popularity scores ('quantile', 'rank' or 'count').

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the LensKitPredictor wrapper of PopScore.
    """
    algo = lenskit_algorithms.create_pop_score(params)
    return LensKitPredictor(algo, name, params, **kwargs)


def create_user_user(name: str, params: Dict[str, Any], **kwargs) -> LensKitPredictor:
    """Create the UserUser predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_neighbors(int): the maximum number of neighbors for scoring each item.
            min_neighbors(int): the minimum number of neighbors for scoring each item.
            min_similarity(float): minimum similarity threshold for considering a neighbor.
            feedback(str): control how feedback should be interpreted ('explicit' or 'implicit').

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the LensKitPredictor wrapper of UserUser.
    """
    algo = lenskit_algorithms.create_user_user(params)
    return LensKitPredictor(algo, name, params, **kwargs)
