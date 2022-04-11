import enum


class RecType(enum.Enum):
    Prediction = 'prediction'
    Recommendation = 'recommendation'


class MetricLibrary(enum.Enum):
    Lenskit = 'Lenskit'
    Rexmex = 'Rexmex'


class MetricCategory(enum.Enum):
    Accuracy = 'Accuracy'
    Rating = 'Rating'
    Coverage = 'Coverage'
    Diversity = 'Diversity'
    Novelty = 'Novelty'


# Metrics
class Metric(enum.Enum):
    # Accuracy
    ndcg = 'NDCG@K'
    precision = 'P@K'
    recall = 'R@K'
    mrr = 'MRR'

    # Rating
    rmse = 'RMSE'
    mae = 'MAE'

    # Coverage
    item_coverage = 'Item Coverage'
    user_coverage = 'User Coverage'

    # Diversity
    gini = 'Gini Index'
    intra_list_similarity = 'Intra-List Similarity'
    similarity_cos = 'Similarity Cosine'
    similarity_euclid = 'Similarity Euclidean'

    # Novelty
    novelty = 'Novelty'

    # ??
    dir = 'DIR'
    pairwise_fairness = 'Pairwise Fairness'


def metric_matches_type(metric, type):
    metrics_by_type = {
        RecType.Recommendation: [Metric.ndcg, Metric.precision, Metric.recall, Metric.mrr],
        RecType.Prediction: [Metric.rmse, Metric.mae]
    }

    # If it's in neither list, it means it can be used for both types
    generic = not (metric in metrics_by_type[RecType.Prediction] or metric in metrics_by_type[RecType.Recommendation])
    return generic or metric in metrics_by_type[type]
