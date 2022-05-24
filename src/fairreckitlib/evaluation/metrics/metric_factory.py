"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from .lenskit.lenskit_predictor import LensKitPredictionEvaluator
from ...core.config_params import ConfigParameters
from ...core.factories import create_factory_from_list
from .common import Metric, MetricCategory

KEY_METRIC_PARAM_K = 'K'



def create_accuracy_metric_factory():
    return create_factory_from_list(MetricCategory.ACCURACY.value, [
        (Metric.NDCG.value,
         create_metric,
         create_metric_params_k
         ),
        (Metric.PRECISION.value,
         create_metric,
         create_metric_params_k
         ),
        (Metric.RECALL.value,
         create_metric,
         create_metric_params_k
         ),
        (Metric.MRR.value,
         create_metric,
         None
         )
    ])


def create_coverage_metric_factory():
    return create_factory_from_list(MetricCategory.COVERAGE.value, [
        (Metric.ITEM_COVERAGE.value,
         create_metric,
         None
         ),
        (Metric.USER_COVERAGE.value,
         create_metric,
         None
         )
    ])


def create_diversity_metric_factory():
    return create_factory_from_list(MetricCategory.DIVERSITY.value, [
        (Metric.INTRA_LIST_SIMILARITY.value,
         create_metric,
         None
         )
    ])


def create_novelty_metric_factory():
    return create_factory_from_list(MetricCategory.NOVELTY.value, [
        (Metric.NOVELTY.value,
         create_metric,
         None
         )
    ])


def create_rating_metric_factory():
    return create_factory_from_list(MetricCategory.RATING.value, [
        (Metric.RMSE.value,
         create_metric,
         None
         ),
        (Metric.MAE.value,
         create_metric,
         None
         )
    ])


def create_metric_params_k():
    params = ConfigParameters()
    params.add_value(KEY_METRIC_PARAM_K, int, None, (1, None))
    return params


def create_metric(name, params, **kwargs):
    # TODO refactor
    from lenskit import topn
    from lenskit.metrics import predict
    from rexmex.metrics import item_coverage, user_coverage, intra_list_similarity, novelty
    from src.fairreckitlib.evaluation.metrics.lenskit.lenskit_recommender import LensKitRecommendationEvaluator
    print('DEV line 92 metric_factory', name, params)
    metric_dict = {
        Metric.NDCG.value: (LensKitRecommendationEvaluator, topn.ndcg, 'ndcg'),
        Metric.PRECISION.value: (LensKitRecommendationEvaluator, topn.precision, 'precision'),
        Metric.RECALL.value: (LensKitRecommendationEvaluator, topn.recall, 'recall'),
        Metric.MRR.value: (LensKitRecommendationEvaluator, topn.recip_rank, 'recip_rank'),

        Metric.RMSE.value: (LensKitPredictionEvaluator, predict.rmse, 'rmse'),
        Metric.MAE.value: (LensKitPredictionEvaluator, predict.mae, 'mae'),

        Metric.ITEM_COVERAGE.value: (LensKitPredictionEvaluator, item_coverage),
        Metric.USER_COVERAGE.value: (LensKitPredictionEvaluator, user_coverage),
        Metric.INTRA_LIST_SIMILARITY.value: (LensKitPredictionEvaluator, intra_list_similarity),
        Metric.NOVELTY.value: (LensKitPredictionEvaluator, novelty)
    }

    evaluator, eval_func, group = metric_dict[name]
    # Add group for LensKit
    if group:
        kwargs['group'] = group
    return evaluator(eval_func, params, **kwargs)


def resolve_metric_factory(metric_name, metric_category_factory):
    for _, factory_name in enumerate(metric_category_factory.get_available_names()):
        metric_factory = metric_category_factory.get_factory(factory_name)
        if metric_factory.is_obj_available(metric_name):
            return metric_factory

    return None
