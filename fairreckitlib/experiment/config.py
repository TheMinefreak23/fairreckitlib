"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

import yaml

from ..pipelines.data.pipeline import DatasetConfig
from ..pipelines.model.pipeline import ModelConfig
from .constants import EXP_KEY_NAME
from .constants import EXP_KEY_TYPE
from .constants import EXP_TYPE_PREDICTION
from .constants import EXP_TYPE_RECOMMENDATION
from .constants import EXP_KEY_TOP_K
from .constants import EXP_KEY_DATASETS
from .constants import EXP_KEY_DATASET_NAME
from .constants import EXP_KEY_DATASET_PREFILTERS
from .constants import EXP_KEY_DATASET_RATING_MODIFIER
from .constants import EXP_KEY_DATASET_SPLIT
from .constants import EXP_KEY_DATASET_SPLIT_TEST_RATIO
from .constants import EXP_KEY_DATASET_SPLIT_TYPE
from .constants import EXP_KEY_MODELS
from .constants import EXP_KEY_MODEL_NAME
from .constants import EXP_KEY_MODEL_PARAMS
from .constants import EXP_KEY_EVALUATION

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
    """Loads an experiment configuration from a yml file.

    Args:
        file_path(str): path to the yml file without extension.
    """
    with open(file_path + '.yml', 'r', encoding='utf-8') as yml_file:
        return yaml.safe_load(yml_file)


def save_config_to_yml(file_path, experiment_config):
    """Saves an experiment configuration to a yml file.

    Args:
        file_path(str): path to the yml file without extension.
        experiment_config(ExperimentConfig): the configuration to save.
    """
    experiment_config = experiment_config_to_dict(experiment_config)

    with open(file_path + '.yml', 'w', encoding='utf-8') as yml_file:
        yaml.dump(experiment_config, yml_file)


def experiment_config_to_dict(experiment_config: ExperimentConfig):
    """Converts an experiment configuration to a dictionary.

    Defines the layout of a configuration file for yml support.

    Args:
        experiment_config(ExperimentConfig): the configuration to convert to dictionary.

    Returns:
        (dict): containing the experiment configuration.
    """
    config = {
        EXP_KEY_NAME: experiment_config.name,
        EXP_KEY_TYPE: experiment_config.type,
        EXP_KEY_DATASETS: [],
        EXP_KEY_MODELS: {},
        EXP_KEY_EVALUATION: {}
    }

    if isinstance(experiment_config, RecommenderExperimentConfig):
        config[EXP_KEY_TOP_K] = experiment_config.top_k

    for _, dataset_config in enumerate(experiment_config.datasets):
        config[EXP_KEY_DATASETS].append({
            EXP_KEY_DATASET_NAME: dataset_config.name,
            EXP_KEY_DATASET_PREFILTERS: dataset_config.prefilters,
            EXP_KEY_DATASET_RATING_MODIFIER: dataset_config.rating_modifier,
            EXP_KEY_DATASET_SPLIT: {
                EXP_KEY_DATASET_SPLIT_TEST_RATIO: dataset_config.splitting.test_ratio,
                EXP_KEY_DATASET_SPLIT_TYPE: dataset_config.splitting.type
            }
        })

    for api_name, models in experiment_config.models.items():
        config[EXP_KEY_MODELS][api_name] = []

        for _, model_config in enumerate(models):
            param_config = {}
            for param_name, param_value in model_config.params.items():
                param_config[param_name] = param_value

            config[EXP_KEY_MODELS][api_name].append({
                EXP_KEY_MODEL_NAME: model_config.name,
                EXP_KEY_MODEL_PARAMS: param_config
            })

    return config
