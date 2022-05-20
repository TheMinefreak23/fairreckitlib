"""This module contains the experiment configurations.

Classes:

    ExperimentConfig: base experiment configuration.
    PredictorExperimentConfig: a prediction experiment configuration.
    RecommenderExperimentConfig: a recommender experiment configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.config_constants import KEY_NAME, KEY_TYPE
from ..core.config_constants import KEY_TOP_K, KEY_RATED_ITEMS_FILTER
from ..core.config_object import object_config_list_to_yml_format
from ..data.data_factory import KEY_DATASETS
from ..data.pipeline.data_config import DatasetConfig
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..evaluation.pipeline.evaluation_config import MetricConfig
from ..model.model_factory import KEY_MODELS
from ..model.pipeline.model_config import ModelConfig, api_models_to_yml_format


@dataclass
class ExperimentConfig:
    """Base Experiment Configuration."""

    datasets: [DatasetConfig]
    models: Dict[str, List[ModelConfig]]
    evaluation: [MetricConfig]
    name: str
    type: str

    def to_yml_format(self) -> Dict[str, Any]:
        """Format experiment configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the experiment configuration.
        """
        yml_format = {
            KEY_NAME: self.name,
            KEY_TYPE: self.type,
            KEY_DATASETS: object_config_list_to_yml_format(self.datasets),
            KEY_MODELS: api_models_to_yml_format(self.models)
        }

        # only include evaluation if it is present
        if len(self.evaluation) > 0:
            yml_format[KEY_EVALUATION] = object_config_list_to_yml_format(self.evaluation)

        return yml_format


@dataclass
class PredictorExperimentConfig(ExperimentConfig):
    """Prediction Experiment Configuration."""


@dataclass
class RecommenderExperimentConfig(ExperimentConfig):
    """Recommender Experiment Configuration."""

    top_k: int
    rated_items_filter: bool

    def to_yml_format(self) -> Dict[str, Any]:
        """Format recommender experiment configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the experiment configuration.
        """
        yml_format = ExperimentConfig.to_yml_format(self)
        yml_format[KEY_TOP_K] = self.top_k
        yml_format[KEY_RATED_ITEMS_FILTER] = self.rated_items_filter
        return yml_format
