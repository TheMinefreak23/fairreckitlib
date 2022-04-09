import enum


class MetricLibrary(enum.Enum):
    Lenskit = 'Lenskit'
    Rexmex = 'Rexmex'


class MainCategory(enum.Enum):
    Prediction = 'Prediction'
    Recommendation = 'Recommendation'


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
