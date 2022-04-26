"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.events import config_event
from fairreckitlib.experiment.parsing import assertion
from ..config import VALID_EXPERIMENT_TYPES
from ..config import PredictorExperimentConfig
from ..config import RecommenderExperimentConfig
from ..config import load_config_from_yml
from ..constants import EXP_KEY_DATASETS
from ..constants import EXP_KEY_MODELS
from ..constants import EXP_KEY_NAME
from ..constants import EXP_KEY_TYPE
from ..constants import EXP_TYPE_PREDICTION
from ..constants import EXP_TYPE_RECOMMENDATION
from ..constants import EXP_KEY_TOP_K
from .datasets import parse_data_config
from .models import parse_models_config


def parse_experiment_config(experiment_config, recommender_system):
    """Parses an experiment configuration.

    Args:
        experiment_config(dict): the experiment's total configuration.
        recommender_system(RecommenderSystem): the system to use for parsing.

    Returns:
        parsed_config(ExperimentConfig): the parsed configuration or None on failure.
    """
    experiment_type = parse_experiment_type(experiment_config, recommender_system.event_dispatcher)

    if experiment_type == EXP_TYPE_PREDICTION:
        return parse_prediction_experiment_config(
            experiment_config,
            recommender_system.data_registry,
            recommender_system.split_factory,
            recommender_system.predictor_factory,
            recommender_system.event_dispatcher
        )
    if experiment_type == EXP_TYPE_RECOMMENDATION:
        return parse_recommender_experiment_config(
            experiment_config,
            recommender_system.data_registry,
            recommender_system.split_factory,
            recommender_system.recommender_factory,
            recommender_system.event_dispatcher
        )

    return None


def parse_experiment_config_from_yml(file_path, recommender_system):
    """Parses an experiment configuration from a yml file.

    Args:
        file_path(str): path to the yml file without extension.
        recommender_system(RecommenderSystem): the system to use for parsing.

    Returns:
        parsed_config(ExperimentConfig): the parsed configuration or None on failure.
    """
    experiment_config = load_config_from_yml(file_path)

    return parse_experiment_config(experiment_config, recommender_system)


def parse_prediction_experiment_config(experiment_config, data_registry, split_factory,
                                       predictor_factory, event_dispatcher):
    """Parses a prediction experiment configuration.

    Args:
        experiment_config(dict): the experiment's total configuration.
        data_registry(DataRegistry): the data registry containing the available datasets.
        split_factory(dict): the split factory containing available splitters.
        predictor_factory(ModelFactory): the model factory containing the available predictors.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(PredictorExperimentConfig): the parsed configuration or None on failure.
    """
    if not parse_experiment_name(experiment_config, event_dispatcher):
        return None

    experiment_name = experiment_config[EXP_KEY_NAME]
    experiment_type = experiment_config[EXP_KEY_TYPE]

    experiment_datasets = parse_experiment_datasets(
        experiment_config,
        data_registry,
        split_factory,
        event_dispatcher
    )
    if experiment_datasets is None:
        return None

    experiment_models = parse_experiment_models(
        experiment_config,
        predictor_factory,
        event_dispatcher
    )
    if experiment_models is None:
        return None

    experiment_evaluation = parse_experiment_evaluation(
        experiment_config,
        event_dispatcher
    )

    parsed_config = PredictorExperimentConfig(
        experiment_datasets,
        experiment_models,
        experiment_evaluation,
        experiment_name,
        experiment_type
    )

    return parsed_config


def parse_recommender_experiment_config(experiment_config, data_registry, split_factory,
                                        recommender_factory, event_dispatcher):
    """Parses a recommender experiment configuration.

    Args:
        experiment_config(dict): the experiment's total configuration.
        data_registry(DataRegistry): the data registry containing the available datasets.
        split_factory(dict): the split factory containing available splitters.
        recommender_factory(ModelFactory): the model factory containing the available recommenders.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        parsed_config(RecommenderExperimentConfig): the parsed configuration or None on failure.
    """
    if not parse_experiment_name(experiment_config, event_dispatcher):
        return None

    experiment_name = experiment_config[EXP_KEY_NAME]
    experiment_type = experiment_config[EXP_KEY_TYPE]
    experiment_top_k = parse_experiment_top_k(experiment_config, 10, event_dispatcher)

    experiment_datasets = parse_experiment_datasets(
        experiment_config,
        data_registry,
        split_factory,
        event_dispatcher
    )
    if experiment_datasets is None:
        return None

    experiment_models = parse_experiment_models(
        experiment_config,
        recommender_factory,
        event_dispatcher
    )
    if experiment_models is None:
        return None

    experiment_evaluation = parse_experiment_evaluation(
        experiment_config,
        event_dispatcher
    )

    return RecommenderExperimentConfig(
        experiment_datasets,
        experiment_models,
        experiment_evaluation,
        experiment_name,
        experiment_type,
        experiment_top_k
    )


def parse_experiment_name(experiment_config, event_dispatcher):
    """Parses the name of the experiment.

    Args:
        experiment_config(dict): the experiment's total configuration.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        success(bool): whether the name is available in the configuration.
    """
    if not assertion.is_key_in_dict(
        EXP_KEY_NAME,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + EXP_KEY_NAME + '\' (required)'
    ): return False

    if not assertion.is_type(
        experiment_config[EXP_KEY_NAME],
        str,
        event_dispatcher,
        'PARSE ERROR: invalid value for experiment key \'' + EXP_KEY_NAME + '\''
    ): return False

    return True


def parse_experiment_datasets(experiment_config, data_registry, split_factory, event_dispatcher):
    """Parses the datasets of the experiment.

    Args:
        experiment_config(dict): the experiment's total configuration.
        data_registry(DataRegistry): the data registry containing the available datasets.
        split_factory(dict): the split factory containing available splitters.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        experiment_datasets(array like): list of parsed DatasetConfig's or None on failure.
    """
    experiment_datasets = parse_data_config(
        experiment_config,
        data_registry,
        split_factory,
        event_dispatcher
    )

    if not assertion.is_container_not_empty(
        experiment_datasets,
        event_dispatcher,
        'PARSE ERROR: no experiment ' + EXP_KEY_DATASETS + ' specified'
    ): return None

    return experiment_datasets


def parse_experiment_evaluation(experiment_config, event_dispatcher):
    # TODO parse experiment evaluation config
    return []


def parse_experiment_models(experiment_config, model_factory, event_dispatcher):
    """Parses the models of the experiment.

    Args:
        experiment_config(dict): the experiment's total configuration.
        model_factory(ModelFactory): the model factory containing the available models.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        experiment_models(dict): dictionary of parsed ModelConfig's keyed by API name
            or None on failure.
    """
    experiment_models = parse_models_config(
        experiment_config,
        model_factory,
        event_dispatcher
    )

    if not assertion.is_container_not_empty(
        experiment_models,
        event_dispatcher,
        'PARSE ERROR: no experiment ' + EXP_KEY_MODELS + ' specified'
    ): return None

    return experiment_models


def parse_experiment_top_k(recommender_experiment_config, default_top_k, event_dispatcher):
    """Parses the top K of the recommender experiment.

    Args:
        recommender_experiment_config(dict): the experiment's total configuration.
        default_top_k(int): the default top K to return on parse failure.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        top_k(int): the topK of the experiment or default_top_k on failure.
    """
    top_k = default_top_k

    if recommender_experiment_config.get(EXP_KEY_TOP_K) is None:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg='PARSE WARNING: missing experiment key \'' + EXP_KEY_TOP_K + '\'',
            default=top_k
        )
    else:
        top_k = recommender_experiment_config[EXP_KEY_TOP_K]

    return top_k


def parse_experiment_type(experiment_config, event_dispatcher):
    """Parses the type of the experiment.

    Args:
        experiment_config(dict): the experiment's total configuration.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.

    Returns:
        experiment_type(str): the type of the experiment or None on failure.
    """
    # assert experiment_config is a dict
    if not assertion.is_type(
        experiment_config,
        dict,
        event_dispatcher,
        'PARSE ERROR: invalid experiment config type'
    ): return None

    # assert EXP_KEY_TYPE is present
    if not assertion.is_key_in_dict(
        EXP_KEY_TYPE,
        experiment_config,
        event_dispatcher,
        'PARSE ERROR: missing experiment key \'' + EXP_KEY_TYPE + '\' (required)',
        one_of_list=VALID_EXPERIMENT_TYPES
    ): return None

    experiment_type = experiment_config[EXP_KEY_TYPE]

    # assert experiment_type is valid
    if not assertion.is_one_of_list(
        experiment_type,
        VALID_EXPERIMENT_TYPES,
        event_dispatcher,
        'PARSE ERROR: invalid value for experiment ' + EXP_KEY_TYPE +
        ' \'' + str(experiment_type) + '\''
    ): return None

    return experiment_type
