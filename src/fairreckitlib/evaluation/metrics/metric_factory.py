"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from typing import Any, Dict

from lenskit import topn
from lenskit.metrics import predict
from rexmex.metrics import item_coverage, user_coverage, intra_list_similarity, novelty

from .evaluator import Evaluator
from .lenskit.lenskit_prediction_evaluator import LensKitPredictionEvaluator
from .lenskit.lenskit_recommendation_evaluator import LensKitRecommendationEvaluator
from .rexmex.rexmex_evaluator import RexmexEvaluator
from ...core.config_params import ConfigParameters
from ...core.factories import create_factory_from_list
from .common import Metric, MetricCategory, KEY_METRIC_PARAM_K


def create_accuracy_metric_factory():
    """Create the factory with Accuracy metrics.

    Returns:
        the factory with all available metrics.
    """
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
    """Create the factory with Coverage metrics.

        Returns:
            the factory with all available metrics.
        """
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


def prepare_for_coverage(self):
    """
    Uses the train set and recommendations set to
    create a tuple that describes the possible user-item pairs in the train set
    and the user-item pairs in the result

    :return: a tuple containing (A,B) where
        a is a tuple (List,List) of possible users and items in the train set and
        b is a list of user-item tuples in the recommendations
    """
    # Convert recommended user, item columns to list of tuples.
    tuple_recs = [tuple(r) for r in self.recs[['source_id', 'target_id']].to_numpy()]
    #print(tuple_recs[0:10])
    # The possible users and items are in the train set
    possible_users_items = (self.train['source_id'], self.train['target_id'])
    #print(possible_users_items)
    return possible_users_items, tuple_recs


def create_diversity_metric_factory():
    """Create the factory with Diversity metrics.

        Returns:
            the factory with all available metrics.
    """
    return create_factory_from_list(MetricCategory.DIVERSITY.value, [
        (Metric.INTRA_LIST_SIMILARITY.value,
         create_metric,
         None
         )
    ])


def create_novelty_metric_factory():
    """Create the factory with Novelty metrics.

        Returns:
            the factory with all available metrics.
    """
    return create_factory_from_list(MetricCategory.NOVELTY.value, [
        (Metric.NOVELTY.value,
         create_metric,
         None
         )
    ])


def create_rating_metric_factory():
    """Create the factory with Rating metrics.

        Returns:
            the factory with all available metrics.
    """
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
    """Create the K param for K metrics

    Returns:
        the configuration parameters of the metric.
    """
    params = ConfigParameters()
    params.add_value(KEY_METRIC_PARAM_K, int, None, (1, None))
    return params


def create_metric(name: str, params: Dict[str, Any], **kwargs) -> Evaluator:
    """Create a metric.

        Args:
            name: the name of the metric.
            params: parameters for the metric
        Returns:
            an Evaluator that uses the metric function corresponding to the name.
    """
    # TODO refactor
    #print('DEV metric name and params', name, params)
    metric_dict = {
        Metric.NDCG.value: (LensKitRecommendationEvaluator, topn.ndcg, 'ndcg'),
        Metric.PRECISION.value: (LensKitRecommendationEvaluator, topn.precision, 'precision'),
        Metric.RECALL.value: (LensKitRecommendationEvaluator, topn.recall, 'recall'),
        Metric.MRR.value: (LensKitRecommendationEvaluator, topn.recip_rank, 'recip_rank'),

        Metric.RMSE.value: (LensKitPredictionEvaluator, predict.rmse, 'rmse'),
        Metric.MAE.value: (LensKitPredictionEvaluator, predict.mae, 'mae'),

        Metric.ITEM_COVERAGE.value: (RexmexEvaluator, item_coverage, None),
        Metric.USER_COVERAGE.value: (RexmexEvaluator, user_coverage, None),
        Metric.INTRA_LIST_SIMILARITY.value: (RexmexEvaluator, intra_list_similarity, None),
        Metric.NOVELTY.value: (RexmexEvaluator, novelty, None)
    }

    evaluator, eval_func, group = metric_dict[name]
    # Add group for LensKit
    if group:
        kwargs['group'] = group
    return evaluator(eval_func, params, **kwargs)


def resolve_metric_factory(metric_name, metric_category_factory):
    """Get metric factory from name and category factory.

    Args:
        metric_name: name of the metric
        metric_category_factory: factory of the metric category

    Returns:
        the metric factory.
    """
    for _, factory_name in enumerate(metric_category_factory.get_available_names()):
        metric_factory = metric_category_factory.get_factory(factory_name)
        if metric_factory.is_obj_available(metric_name):
            return metric_factory

    return None
