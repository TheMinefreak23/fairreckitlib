"""This module contains functionality to create the surprise predictor/recommender factory.

Functions:

    create_predictor_factory: create factory with surprise predictors.
    create_recommender_factory: create factory with surprise recommenders.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.apis import SURPRISE_API
from ....core.factories import Factory, create_factory_from_list
from . import surprise_params
from . import surprise_predictor
from . import surprise_recommender
from . import surprise_algorithms


def create_predictor_factory() -> Factory:
    """Create the factory with Surprise predictors.

    Returns:
        the factory with all available predictors.
    """
    return create_factory_from_list(SURPRISE_API, [
        (surprise_algorithms.BASELINE_ONLY_ALS,
         surprise_predictor.create_baseline_only_als,
         surprise_params.create_params_baseline_only_als
         ),
        (surprise_algorithms.BASELINE_ONLY_SGD,
         surprise_predictor.create_baseline_only_sgd,
         surprise_params.create_params_baseline_only_sgd
         ),
        (surprise_algorithms.CO_CLUSTERING,
         surprise_predictor.create_co_clustering,
         surprise_params.create_params_co_clustering
         ),
        (surprise_algorithms.KNN_BASIC,
         surprise_predictor.create_knn_basic,
         surprise_params.create_params_knn_similarities
         ),
        (surprise_algorithms.KNN_BASELINE_ALS,
         surprise_predictor.create_knn_baseline_als,
         surprise_params.create_params_knn_baseline_als
         ),
        (surprise_algorithms.KNN_BASELINE_SGD,
         surprise_predictor.create_knn_baseline_sgd,
         surprise_params.create_params_knn_baseline_sgd
         ),
        (surprise_algorithms.KNN_WITH_MEANS,
         surprise_predictor.create_knn_with_means,
         surprise_params.create_params_knn_similarities
         ),
        (surprise_algorithms.KNN_WITH_ZSCORE,
         surprise_predictor.create_knn_with_zscore,
         surprise_params.create_params_knn_similarities
         ),
        (surprise_algorithms.NMF,
         surprise_predictor.create_nmf,
         surprise_params.create_params_nmf
         ),
        (surprise_algorithms.NORMAL_PREDICTOR,
         surprise_predictor.create_normal_predictor,
         None
         ),
        (surprise_algorithms.SLOPE_ONE,
         surprise_predictor.create_slope_one,
         None
         ),
        (surprise_algorithms.SVD,
         surprise_predictor.create_svd,
         surprise_params.create_params_svd
         ),
        (surprise_algorithms.SVD_PP,
         surprise_predictor.create_svd_pp,
         surprise_params.create_params_svd_pp
         )
    ])


def create_recommender_factory() -> Factory:
    """Create the factory with Surprise recommenders.

    Returns:
        the factory with all available recommenders.
    """
    return create_factory_from_list(SURPRISE_API, [
        (surprise_algorithms.BASELINE_ONLY_ALS,
         surprise_recommender.create_baseline_only_als,
         surprise_params.create_params_baseline_only_als
         ),
        (surprise_algorithms.BASELINE_ONLY_SGD,
         surprise_recommender.create_baseline_only_sgd,
         surprise_params.create_params_baseline_only_sgd
         ),
        (surprise_algorithms.CO_CLUSTERING,
         surprise_recommender.create_co_clustering,
         surprise_params.create_params_co_clustering
         ),
        (surprise_algorithms.KNN_BASIC,
         surprise_recommender.create_knn_basic,
         surprise_params.create_params_knn_similarities
         ),
        (surprise_algorithms.KNN_BASELINE_ALS,
         surprise_recommender.create_knn_baseline_als,
         surprise_params.create_params_knn_baseline_als
         ),
        (surprise_algorithms.KNN_BASELINE_SGD,
         surprise_recommender.create_knn_baseline_sgd,
         surprise_params.create_params_knn_baseline_sgd
         ),
        (surprise_algorithms.KNN_WITH_MEANS,
         surprise_recommender.create_knn_with_means,
         surprise_params.create_params_knn_similarities
         ),
        (surprise_algorithms.KNN_WITH_ZSCORE,
         surprise_recommender.create_knn_with_zscore,
         surprise_params.create_params_knn_similarities
         ),
        (surprise_algorithms.NMF,
         surprise_recommender.create_nmf,
         surprise_params.create_params_nmf
         ),
        (surprise_algorithms.NORMAL_PREDICTOR,
         surprise_recommender.create_normal_predictor,
         None
         ),
        (surprise_algorithms.SLOPE_ONE,
         surprise_recommender.create_slope_one,
         None
         ),
        (surprise_algorithms.SVD,
         surprise_recommender.create_svd,
         surprise_params.create_params_svd
         ),
        (surprise_algorithms.SVD_PP,
         surprise_recommender.create_svd_pp,
         surprise_params.create_params_svd_pp
         )
    ])
