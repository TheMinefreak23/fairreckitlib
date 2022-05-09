"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..experiment.constants import EXP_KEY_METRIC_PARAM_K
from src.fairreckitlib.core.params import ConfigParameters
from src.fairreckitlib.core.params import get_empty_parameters
from .common import Metric


class MetricFactory:

    def __init__(self):
        self.prediction_metrics = [Metric.RMSE, Metric.MAE]
        self.recommendation_metrics = [Metric.NDCG, Metric.PRECISION, Metric.RECALL, Metric.MRR]

    def get_params(self, metric_name):
        if not metric_name in [Metric.NDCG.value, Metric.PRECISION.value, Metric.RECALL.value]:
            return get_empty_parameters()

        params = ConfigParameters()
        params.add_value(EXP_KEY_METRIC_PARAM_K, int, None, (1, None))
        return params

    def get_available_prediction_metric_names(self):
        names = []

        for _, metric in enumerate(self.prediction_metrics):
            names.append(metric.value)

        return names

    def get_available_recommendation_metric_names(self):
        names = []

        for _, metric in enumerate(self.recommendation_metrics):
            names.append(metric.value)

        return names


def create_metric_factory():
    return MetricFactory()
