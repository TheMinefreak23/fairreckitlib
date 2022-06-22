"""This module contains the experiment configurations.

Classes:

    ExperimentConfig: base experiment configuration.
    PredictorExperimentConfig: a prediction experiment configuration.
    RecommenderExperimentConfig: a recommender experiment configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.config.config_yml import format_yml_config_dict_list, format_yml_config_list
from ..core.core_constants import KEY_NAME, KEY_TYPE, KEY_TOP_K, KEY_RATED_ITEMS_FILTER, \
    TYPE_PREDICTION, TYPE_RECOMMENDATION
from ..data.data_factory import KEY_DATA
from ..data.pipeline.data_config import DataMatrixConfig
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..evaluation.pipeline.evaluation_config import MetricConfig
from ..model.model_factory import KEY_MODELS
from ..model.pipeline.model_config import ModelConfig


@dataclass
class ExperimentConfig(metaclass=ABCMeta):
    """Base Experiment Configuration."""

    datasets: List[DataMatrixConfig]
    models: Dict[str, List[ModelConfig]]
    evaluation: List[MetricConfig]
    name: str

    @abstractmethod
    def get_type(self) -> str:
        """Get the type of the experiment configuration.

        Returns:
            the experiment type.
        """
        raise NotImplementedError()

    def to_yml_format(self) -> Dict[str, Any]:
        """Format experiment configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the experiment configuration.
        """
        yml_format = {
            KEY_NAME: self.name,
            KEY_TYPE: self.get_type(),
            KEY_DATA: format_yml_config_list(self.datasets),
            KEY_MODELS: format_yml_config_dict_list(self.models)
        }

        # only include evaluation if it is present
        if len(self.evaluation) > 0:
            yml_format[KEY_EVALUATION] = format_yml_config_list(self.evaluation)

        return yml_format


@dataclass
class PredictorExperimentConfig(ExperimentConfig):
    """Prediction Experiment Configuration."""

    def get_type(self) -> str:
        """Get the predictor experiment type.

        Returns:
            TYPE_PREDICTION.
        """
        return TYPE_PREDICTION


@dataclass
class RecommenderExperimentConfig(ExperimentConfig):
    """Recommender Experiment Configuration."""

    top_k: int
    rated_items_filter: bool

    def get_type(self) -> str:
        """Get the recommender experiment type.

        Returns:
            TYPE_RECOMMENDATION.
        """
        return TYPE_RECOMMENDATION

    def to_yml_format(self) -> Dict[str, Any]:
        """Format recommender experiment configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the experiment configuration.
        """
        yml_format = ExperimentConfig.to_yml_format(self)
        yml_format[KEY_TOP_K] = self.top_k
        yml_format[KEY_RATED_ITEMS_FILTER] = self.rated_items_filter
        return yml_format
