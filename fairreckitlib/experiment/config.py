"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

import yaml

from ..pipelines.data.pipeline import DatasetConfig
from ..pipelines.model.pipeline import ModelConfig
from .constants import EXP_TYPE_PREDICTION
from .constants import EXP_TYPE_RECOMMENDATION

VALID_EXPERIMENT_TYPES = [EXP_TYPE_PREDICTION, EXP_TYPE_RECOMMENDATION]


@dataclass
class ExperimentConfig:
    """Base Experiment Configuration."""

    datasets: [DatasetConfig]
    models: {str: [ModelConfig]}
    evaluation: {}
    name: str
    type: str


@dataclass
class PredictorExperimentConfig(ExperimentConfig):
    """Prediction Experiment Configuration."""


@dataclass
class RecommenderExperimentConfig(ExperimentConfig):
    """Recommender Experiment Configuration."""

    top_k: int


def load_config_from_yml(file_path):
    """Loads a configuration from a yml file.

    Args:
        file_path(str): path to the yml file without extension.
    """
    with open(file_path + '.yml', 'r', encoding='utf-8') as yml_file:
        return yaml.safe_load(yml_file)
