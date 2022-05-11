"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

from ..core.config_constants import KEY_NAME, KEY_PARAMS, KEY_TYPE
from ..core.config_constants import KEY_TOP_K, KEY_RATED_ITEMS_FILTER
from ..data.pipeline.data_config import DatasetConfig, KEY_DATASETS, KEY_DATA_FILTERS
from ..data.ratings.rating_converter_factory import KEY_RATING_MODIFIER
from ..data.split.split_factory import KEY_SPLITTING, KEY_SPLIT_TEST_RATIO
from ..data.utility import save_yml
from ..evaluation.pipeline.evaluation_config import MetricConfig, KEY_EVALUATION
from ..model.pipeline.model_config import ModelConfig, KEY_MODELS


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


def save_config_to_yml(file_path, experiment_config):
    """Saves an experiment configuration to a yml file.

    Args:
        file_path(str): path to the yml file without extension.
        experiment_config(ExperimentConfig): the configuration to save.
    """
    experiment_config = experiment_config_to_dict(experiment_config)
    save_yml(file_path + '.yml', experiment_config)


def experiment_config_to_dict(experiment_config: ExperimentConfig):
    """Converts an experiment configuration to a dictionary.

    Defines the layout of a configuration file for yml support.

    Args:
        experiment_config(ExperimentConfig): the configuration to convert to dictionary.

    Returns:
        (dict): containing the experiment configuration.
    """
    config = {
        KEY_NAME: experiment_config.name,
        KEY_TYPE: experiment_config.type,
        KEY_DATASETS: [],
        KEY_MODELS: {}
    }

    if isinstance(experiment_config, RecommenderExperimentConfig):
        config[KEY_TOP_K] = experiment_config.top_k
        config[KEY_RATED_ITEMS_FILTER] = experiment_config.rated_items_filter

    for _, dataset_config in enumerate(experiment_config.datasets):
        dataset = {
            KEY_NAME: dataset_config.name,
            KEY_SPLITTING: split_config_to_dict(dataset_config.splitting)
        }

        # only include prefilters if it has entries
        if len(dataset_config.prefilters) > 0:
            # TODO
            dataset[KEY_DATA_FILTERS] = []

        # only include rating modifier if it is present
        if dataset_config.rating_modifier:
            # TODO
            dataset[KEY_RATING_MODIFIER] = dataset_config.rating_modifier

        config[KEY_DATASETS].append(dataset)

    for api_name, models in experiment_config.models.items():
        config[KEY_MODELS][api_name] = []

        for _, model_config in enumerate(models):
            param_config = {}
            for param_name, param_value in model_config.params.items():
                param_config[param_name] = param_value

            model = {KEY_NAME: model_config.name}

            # only include model params if it has entries
            if len(param_config) > 0:
                model[KEY_PARAMS] = param_config

            config[KEY_MODELS][api_name].append(model)

    # only include evaluation if it is present
    if len(experiment_config.evaluation) > 0:
        config[KEY_EVALUATION] = []

        for _, metric_config in enumerate(experiment_config.evaluation):
            metric = {KEY_NAME: metric_config.name}

            # only include metric params if it has entries
            if len(metric_config.params) > 0:
                metric[KEY_PARAMS] = metric_config.params

            # only include prefilters if it has entries
            if len(metric_config.prefilters) > 0:
                # TODO
                metric[KEY_DATA_FILTERS] = metric_config.prefilters

            config[KEY_EVALUATION].append(metric)

    return config


def split_config_to_dict(split_config):
    """Converts a splitting configuration to a dictionary.

    Args:
        split_config(SplitConfig): the configuration to convert to dictionary.

    Returns:
        (dict): containing the splitting configuration.
    """
    splitting = {
        KEY_SPLIT_TEST_RATIO: split_config.test_ratio,
        KEY_NAME: split_config.type
    }

    # only include splitting params if it has entries
    if len(split_config.params) > 0:
        splitting[KEY_PARAMS] = split_config.params

    return splitting
