"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List

import numpy as np
import lenskit
from lenskit import batch
from lenskit.algorithms import CandidateSelector
from lenskit.algorithms.basic import AllItemsCandidateSelector, UnratedItemCandidateSelector
from lenskit.algorithms.ranking import TopN
import pandas as pd

from ..base_recommender import Recommender
from . import lenskit_algorithms


class LensKitRecommender(Recommender):
    """Recommender implementation for the LensKit framework."""

    def __init__(self, algo: lenskit.Recommender, name: str, params: Dict[str, Any], **kwargs):
        """Construct the lenskit recommender.

        Args:
            algo: the lenskit recommender algorithm.
            name: the name of the recommender.
            params: the parameters of the recommender.

        Keyword Args:
            num_threads(int): the max number of threads the recommender can use.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.
        """
        Recommender.__init__(self, name, params,
                             kwargs['num_threads'], kwargs['rated_items_filter'])
        self.algo = algo

    def on_train(self) -> None:
        """Fit the lenskit algorithm on the train set."""
        self.algo.fit(self.train_set)

    def on_recommend(self, user: int, num_items: int) -> pd.DataFrame:
        """Compute item recommendations for the specified user.

        Lenskit recommenders have an implementation that is exactly the
        same as the required interface.

        Args:
            user: the user ID to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            dataframe with the columns: 'item' and 'score'.
        """
        recs = self.algo.recommend(user, n=num_items)
        # random algo does not produce a score
        if self.get_name() == lenskit_algorithms.RANDOM:
            recs['score'] = np.full(num_items, 1)

        return recs

    def on_recommend_batch(self, users: List[int], num_items: int) -> pd.DataFrame:
        """Compute the items recommendations for each of the specified users.

        Lenskit recommenders have a batch implementation available that allows for
        recommending items using multiple 'jobs'.

        Args:
            users: the user ID's to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            dataframe with the columns: 'rank', 'user', 'item', 'score'.
        """
        n_jobs = self.num_threads if self.num_threads > 0 else None
        recs = batch.recommend(self.algo, users, num_items, n_jobs=n_jobs)

        # random algo does not produce a score
        if self.get_name() == lenskit_algorithms.RANDOM:
            recs['score'] = np.full(len(users) * num_items, 1)

        return recs[['rank', 'user', 'item', 'score']]


def create_candidate_selector(rated_items_filter: bool) -> CandidateSelector:
    """Create a candidate selector for the specified filter.

    Args:
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the corresponding lenskit candidate selector.
    """
    return UnratedItemCandidateSelector() if rated_items_filter else AllItemsCandidateSelector()


def create_biased_mf(name: str, params: Dict[str, Any], **kwargs) -> LensKitRecommender:
    """Create the BiasedMF recommender.

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
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the LensKitRecommender wrapper of BiasedMF.
    """
    algo = TopN(
        lenskit_algorithms.create_biased_mf(params),
        create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def create_implicit_mf(name: str, params: Dict[str, Any], **kwargs) -> LensKitRecommender:
    """Create the ImplicitMF recommender.

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
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the LensKitRecommender wrapper of ImplicitMF.
    """
    algo = TopN(
        lenskit_algorithms.create_implicit_mf(params),
        create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def create_item_item(name: str, params: Dict[str, Any], **kwargs) -> LensKitRecommender:
    """Create the ItemItem recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_neighbors(int): the maximum number of neighbors for scoring each item.
            min_neighbors(int): the minimum number of neighbors for scoring each item.
            min_similarity(float): minimum similarity threshold for considering a neighbor.
            feedback(str): control how feedback should be interpreted ('explicit' or 'implicit').

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the LensKitRecommender wrapper of ItemItem.
    """
    algo = TopN(
        lenskit_algorithms.create_item_item(params),
        create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def create_pop_score(name: str, params: Dict[str, Any], **kwargs) -> LensKitRecommender:
    """Create the PopScore recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            score_method(str): for computing popularity scores ('quantile', 'rank' or 'count').

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the LensKitRecommender wrapper of PopScore.
    """
    algo = TopN(
        lenskit_algorithms.create_pop_score(params),
        create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def create_random(name: str, params: Dict[str, Any], **kwargs) -> LensKitRecommender:
    """Create the Random recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            random_seed(int): the random seed or None for the current time as seed.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the LensKitRecommender wrapper of Random.
    """
    algo = lenskit_algorithms.create_random(
        params,
        create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def create_user_user(name: str, params: Dict[str, Any], **kwargs) -> LensKitRecommender:
    """Create the UserUser recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_neighbors(int): the maximum number of neighbors for scoring each item.
            min_neighbors(int): the minimum number of neighbors for scoring each item.
            min_similarity(float): minimum similarity threshold for considering a neighbor.
            feedback(str): control how feedback should be interpreted ('explicit' or 'implicit').

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        the LensKitRecommender wrapper of UserUser.
    """
    algo = TopN(
        lenskit_algorithms.create_user_user(params),
        create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)
