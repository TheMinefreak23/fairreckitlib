"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

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
    # TODO metric connection
    raise NotImplementedError()


def resolve_metric_factory(metric_name, metric_category_factory):
    for _, factory_name in enumerate(metric_category_factory.get_available_names()):
        metric_factory = metric_category_factory.get_factory(factory_name)
        if metric_factory.is_obj_available(metric_name):
            return metric_factory

    return None
