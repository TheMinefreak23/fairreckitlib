"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

import yaml

from ..pipelines.data.pipeline import DatasetConfig
from ..pipelines.evaluation.pipeline import MetricConfig
from ..pipelines.model.pipeline import ModelConfig
from .constants import EXP_KEY_NAME
from .constants import EXP_KEY_TYPE
from .constants import EXP_TYPE_PREDICTION
from .constants import EXP_TYPE_RECOMMENDATION
from .constants import EXP_KEY_TOP_K
from .constants import EXP_KEY_RATED_ITEMS_FILTER
from .constants import EXP_KEY_DATASETS
from .constants import EXP_KEY_DATASET_NAME
from .constants import EXP_KEY_DATASET_PREFILTERS
from .constants import EXP_KEY_DATASET_RATING_MODIFIER
from .constants import EXP_KEY_DATASET_SPLIT
from .constants import EXP_KEY_DATASET_SPLIT_PARAMS
from .constants import EXP_KEY_DATASET_SPLIT_TEST_RATIO
from .constants import EXP_KEY_DATASET_SPLIT_TYPE
from .constants import EXP_KEY_MODELS
from .constants import EXP_KEY_MODEL_NAME
from .constants import EXP_KEY_MODEL_PARAMS
from .constants import EXP_KEY_EVALUATION
from .constants import EXP_KEY_METRIC_NAME
from .constants import EXP_KEY_METRIC_PARAMS
from .constants import EXP_KEY_METRIC_PREFILTERS

VALID_EXPERIMENT_TYPES = [EXP_TYPE_PREDICTION, EXP_TYPE_RECOMMENDATION]


@dataclass
class ExperimentConfig:
    """Base Experiment Configuration."""

    datasets: [DatasetConfig]
    models: {str: [ModelConfig]}
    evaluation: [MetricConfig]
    name: str
    type: str


@dataclass
class PredictorExperimentConfig(ExperimentConfig):
    """Prediction Experiment Configuration."""


@dataclass
class RecommenderExperimentConfig(ExperimentConfig):
    """Recommender Experiment Configuration."""

    top_k: int
    rated_items_filter: bool


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
        EXP_KEY_MODELS: {}
    }

    if isinstance(experiment_config, RecommenderExperimentConfig):
        config[EXP_KEY_TOP_K] = experiment_config.top_k
        config[EXP_KEY_RATED_ITEMS_FILTER] = experiment_config.rated_items_filter

    for _, dataset_config in enumerate(experiment_config.datasets):
        dataset = {
            EXP_KEY_DATASET_NAME: dataset_config.name,
            EXP_KEY_DATASET_SPLIT: split_config_to_dict(dataset_config.splitting)
        }

        # only include prefilters if it has entries
        if len(dataset_config.prefilters) > 0:
            # TODO
            dataset[EXP_KEY_DATASET_PREFILTERS] = []

        # only include rating modifier if it is present
        if dataset_config.rating_modifier:
            # TODO
            dataset[EXP_KEY_DATASET_RATING_MODIFIER] = dataset_config.rating_modifier

        config[EXP_KEY_DATASETS].append(dataset)

    for api_name, models in experiment_config.models.items():
        config[EXP_KEY_MODELS][api_name] = []

        for _, model_config in enumerate(models):
            param_config = {}
            for param_name, param_value in model_config.params.items():
                param_config[param_name] = param_value

            model = {EXP_KEY_MODEL_NAME: model_config.name}

            # only include model params if it has entries
            if len(param_config) > 0:
                model[EXP_KEY_MODEL_PARAMS] = param_config

            config[EXP_KEY_MODELS][api_name].append(model)

    # only include evaluation if it is present
    if len(experiment_config.evaluation) > 0:
        config[EXP_KEY_EVALUATION] = []

        for _, metric_config in enumerate(experiment_config.evaluation):
            metric = {EXP_KEY_METRIC_NAME: metric_config.name}

            # only include metric params if it has entries
            if len(metric_config.params) > 0:
                metric[EXP_KEY_METRIC_PARAMS] = metric_config.params

            # only include prefilters if it has entries
            if len(metric_config.prefilters) > 0:
                # TODO
                metric[EXP_KEY_METRIC_PREFILTERS] = metric_config.prefilters

            config[EXP_KEY_EVALUATION].append(metric)

    return config


def split_config_to_dict(split_config):
    """Converts a splitting configuration to a dictionary.

    Args:
        split_config(SplitConfig): the configuration to convert to dictionary.

    Returns:
        (dict): containing the splitting configuration.
    """
    splitting = {
        EXP_KEY_DATASET_SPLIT_TEST_RATIO: split_config.test_ratio,
        EXP_KEY_DATASET_SPLIT_TYPE: split_config.type
    }

    # only include splitting params if it has entries
    if len(split_config.params) > 0:
        splitting[EXP_KEY_DATASET_SPLIT_PARAMS] = split_config.params

    return splitting
