"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import enum


class Test:
    """
    DEV ONLY
    """
    def __init__(self, name, recs_path, test_path, train_path, rec_type):
        self.name = name
        self.recs_path = recs_path
        self.test_path = test_path
        self.train_path = train_path
        self.rec_type = rec_type


class RecType(enum.Enum):
    """
    Type of recommendation
    """
    PREDICTION = 'prediction'
    RECOMMENDATION = 'recommendation'


class MetricLibrary(enum.Enum):
    """
    API/Library used for computing metric
    """
    LENSKIT = 'Lenskit'
    REXMEX = 'Rexmex'


class MetricCategory(enum.Enum):
    """
    Metric category of evaluation
    """
    ACCURACY = 'Accuracy'
    RATING = 'Rating'
    COVERAGE = 'Coverage'
    DIVERSITY = 'Diversity'
    NOVELTY = 'Novelty'


# Known metrics (some not supported)
class Metric(enum.Enum):
    """
    Known metrics (names)
    """
    # Accuracy
    NDCG = 'NDCG@K'
    PRECISION = 'P@K'
    RECALL = 'R@K'
    MRR = 'MRR'

    # Rating
    RMSE = 'RMSE'
    MAE = 'MAE'

    # Coverage
    ITEM_COVERAGE = 'Item Coverage'
    USER_COVERAGE = 'User Coverage'

    # Diversity
    GINI = 'Gini Index'
    INTRA_LIST_SIMILARITY = 'Intra-List Similarity'
    SIMILARITY_COS = 'Similarity Cosine'
    SIMILARITY_EUCLID = 'Similarity Euclidean'

    # Novelty
    NOVELTY = 'Novelty'

    # ??
    DIR = 'DIR'
    PAIRWISE_FAIRNESS = 'Pairwise Fairness'


# TODO metric classes
metric_category_dict = {
    MetricCategory.Accuracy: [Metric.ndcg, Metric.precision, Metric.recall, Metric.mrr],
    MetricCategory.Rating: [Metric.rmse, Metric.mae],
    MetricCategory.Coverage: [Metric.item_coverage, Metric.user_coverage],
    MetricCategory.Diversity: [Metric.gini, Metric.intra_list_similarity, Metric.similarity_cos,
                               Metric.similarity_euclid],
    MetricCategory.Novelty: [Metric.novelty]
}


def metric_matches_type(metric, rec_type):
    """
    Check whether the metric can be used on the recommendation result type

    :param metric: the metric to use for analysis
    :param rec_type: the type of the recommenaation result
    :return: a Boolean which specifies whether the metric can be used for the result
    """
    metrics_by_type = {
        RecType.RECOMMENDATION: [Metric.NDCG, Metric.PRECISION, Metric.RECALL, Metric.MRR],
        RecType.PREDICTION: [Metric.RMSE, Metric.MAE]
    }

    # If it's in neither list, it means it can be used for both types
    is_predictor = metric in metrics_by_type[RecType.PREDICTION]
    is_recommender = metric in metrics_by_type[RecType.RECOMMENDATION]
    generic = not is_predictor or is_recommender
    return generic or metric in metrics_by_type[rec_type]
