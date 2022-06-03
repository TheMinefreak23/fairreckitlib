"""This module contains functionality to create and resolve metric factories.

Functions:

    create_metric_params_k: create metric config parameters with a K param.
    create_accuracy_metric_factory: create metric category factory for accuracy metrics.
    create_coverage_metric_factory: create metric category factory for coverage metrics.
    create_rating_metric_factory: create metric category factory for rating metrics.
    resolve_metric_factory: resolve the metric factory from the name of the metric.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Optional

from ...core.config.config_factories import Factory, GroupFactory, create_factory_from_list
from ...core.config.config_parameters import ConfigParameters
from .lenskit import lenskit_accuracy_metric, lenskit_rating_metric
from .rexmex import rexmex_coverage_metric, rexmex_rating_metric
from .metric_constants import KEY_METRIC_PARAM_K, MetricCategory, Metric


def create_metric_params_k() -> ConfigParameters:
    """Create the parameters of a metric that utilizes the K param.

    Returns:
        the configuration parameters of the metric.
    """
    params = ConfigParameters()
    params.add_number(KEY_METRIC_PARAM_K, int, None, (1, None))
    return params


def create_accuracy_metric_factory() -> Factory:
    """Create the factory with Accuracy metrics.

    Returns:
        the factory with all available metrics.
    """
    return create_factory_from_list(MetricCategory.ACCURACY.value, [
        (Metric.HIT_RATIO.value, lenskit_accuracy_metric.create_hit_ratio, create_metric_params_k),
        (Metric.NDCG.value, lenskit_accuracy_metric.create_ndcg, create_metric_params_k),
        (Metric.PRECISION.value, lenskit_accuracy_metric.create_precision, create_metric_params_k),
        (Metric.RECALL.value, lenskit_accuracy_metric.create_recall, create_metric_params_k),
        (Metric.MRR.value, lenskit_accuracy_metric.create_mean_recip_rank, None),
    ])


def create_coverage_metric_factory() -> Factory:
    """Create the factory with Coverage metrics.

    Returns:
        the factory with all available metrics.
    """
    return create_factory_from_list(MetricCategory.COVERAGE.value, [
        (Metric.ITEM_COVERAGE.value, rexmex_coverage_metric.create_item_coverage, None),
        (Metric.USER_COVERAGE.value, rexmex_coverage_metric.create_user_coverage, None)
    ])


def create_rating_metric_factory() -> Factory:
    """Create the factory with Rating metrics.

    Returns:
        the factory with all available metrics.
    """
    return create_factory_from_list(MetricCategory.RATING.value, [
        (Metric.MAE.value, lenskit_rating_metric.create_mae, None),
        (Metric.MAPE.value, rexmex_rating_metric.create_mape, None),
        (Metric.MSE.value, rexmex_rating_metric.create_mse, None),
        (Metric.RMSE.value, lenskit_rating_metric.create_rmse, None)
    ])


def resolve_metric_factory(
        metric_name: str,
        metric_category_factory: GroupFactory) -> Optional[Factory]:
    """Resolve the metric factory from the name of the metric.

    Args:
        metric_name: the name of the metric.
        metric_category_factory: the factory that contains metric category factories.

    Returns:
        the resolved metric factory or None when not found.
    """
    for factory_name in metric_category_factory.get_available_names():
        metric_factory = metric_category_factory.get_factory(factory_name)
        if metric_factory.is_obj_available(metric_name):
            return metric_factory

    return None
