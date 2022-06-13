"""This module contains metric constants that are used in other modules.

Constants:

    KEY_METRIC_PARAM_K: the key that is used for the metric parameter K.
    KEY_METRIC_SUBGROUP: the key that is used to identify a metric subgroup.

Enumerations:

    MetricCategory: the category of evaluation for a metric.
    Metric: the names of various metrics.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import enum

KEY_METRIC_PARAM_K = 'K'
KEY_METRIC_SUBGROUP = 'subgroup'


class MetricCategory(enum.Enum):
    """Metric category of evaluation."""

    ACCURACY = 'Accuracy'
    RATING = 'Rating'
    COVERAGE = 'Coverage'
    DIVERSITY = 'Diversity'
    NOVELTY = 'Novelty'


# Known metrics (some not supported)
class Metric(enum.Enum):
    """Known metrics (names)."""

    # Accuracy
    HIT_RATIO = 'HR@K'
    NDCG = 'NDCG@K'
    PRECISION = 'P@K'
    RECALL = 'R@K'
    MRR = 'MRR'

    # Coverage
    ITEM_COVERAGE = 'Item Coverage'
    USER_COVERAGE = 'User Coverage'

    # Diversity
    GINI = 'Gini Index'
    INTRA_LIST_SIMILARITY = 'Intra-List Similarity'
    SIMILARITY_COS = 'Similarity Cosine'
    SIMILARITY_EUCLID = 'Similarity Euclidean'

    # Rating
    RMSE = 'RMSE'
    MAE = 'MAE'
    MAPE = 'MAPE'
    MSE = 'MSE'
