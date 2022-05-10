"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.config_constants import KEY_TYPE, TYPE_PREDICTION, TYPE_RECOMMENDATION, VALID_TYPES
from ..core.config_constants import KEY_NAME, KEY_TOP_K, DEFAULT_TOP_K
from ..core.config_constants import KEY_RATED_ITEMS_FILTER, DEFAULT_RATED_ITEMS_FILTER
from ..core.config_params import ConfigOptionParam, ConfigValueParam
from ..core.event_dispatcher import EventDispatcher
from ..core.parsing.parse_assert import assert_is_container_not_empty, assert_is_type
from ..core.parsing.parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from ..core.parsing.parse_event import ON_PARSE, on_parse
from ..core.parsing.parse_params import parse_config_param
from ..data.pipeline.data_config import KEY_DATASETS
from ..data.pipeline.data_config_parsing import parse_data_config
from ..data.utility import load_yml
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..evaluation.pipeline.evaluation_config_parsing import parse_evaluation_config
from ..model.model_factory import KEY_MODELS
from ..model.pipeline.model_config_parsing import parse_models_config
from .experiment_config import PredictorExperimentConfig, RecommenderExperimentConfig


class Parser:
    def __init__(self, verbose):
        self.verbose = verbose
        self.event_dispatcher = EventDispatcher()
        self.event_dispatcher.add_listener(ON_PARSE, self, (on_parse, None))

    def parse_experiment_config(self, experiment_config, data_registry, experiment_factory):
        """Parses an experiment configuration.

        Args:
            experiment_config(dict): the experiment's total configuration.
            data_registry(DataRegistry): the data registry containing the available datasets.
            experiment_factory(GroupFactory): the factory containing all three pipeline factories.

        Returns:
            parsed_config(ExperimentConfig): the parsed configuration or None on failure.
        """
        experiment_type = self.parse_experiment_type(experiment_config)

        if experiment_type == TYPE_PREDICTION:
            return self.parse_prediction_experiment_config(
                experiment_config,
                data_registry,
                experiment_factory
            )
        if experiment_type == TYPE_RECOMMENDATION:
            return self.parse_recommender_experiment_config(
                experiment_config,
                data_registry,
                experiment_factory
            )

        return None

    def parse_experiment_config_from_yml(self, file_path, data_registry, experiment_factory):
        """Parses an experiment configuration from a yml file.

        Args:
            file_path(str): path to the yml file without extension.
            data_registry(DataRegistry): the data registry containing the available datasets.
            experiment_factory(GroupFactory): the factory containing all three pipeline factories.

        Returns:
            parsed_config(ExperimentConfig): the parsed configuration or None on failure.
        """
        experiment_config = load_yml(file_path + '.yml')

        return self.parse_experiment_config(experiment_config, data_registry, experiment_factory)

    def parse_prediction_experiment_config(self, experiment_config, data_registry, experiment_factory):
        """Parses a prediction experiment configuration.

        Args:
            experiment_config(dict): the experiment's total configuration.
            data_registry(DataRegistry): the data registry containing the available datasets.
            experiment_factory(GroupFactory): the factory containing all three pipeline factories.

        Returns:
            parsed_config(PredictorExperimentConfig): the parsed configuration or None on failure.
        """
        if not self.parse_experiment_name(experiment_config):
            return None

        experiment_name = experiment_config[KEY_NAME]
        experiment_type = experiment_config[KEY_TYPE]

        experiment_datasets = self.parse_experiment_datasets(
            experiment_config,
            data_registry,
            experiment_factory.get_factory(KEY_DATASETS)
        )
        if experiment_datasets is None:
            return None

        experiment_models = self.parse_experiment_models(
            experiment_config,
            experiment_factory.get_factory(KEY_MODELS).get_factory(TYPE_PREDICTION)
        )
        if experiment_models is None:
            return None

        experiment_evaluation = self.parse_experiment_evaluation(
            experiment_config,
            experiment_factory.get_factory(KEY_EVALUATION).get_factory(TYPE_PREDICTION)
        )

        parsed_config = PredictorExperimentConfig(
            experiment_datasets,
            experiment_models,
            experiment_evaluation,
            experiment_name,
            experiment_type
        )

        return parsed_config

    def parse_recommender_experiment_config(self, experiment_config, data_registry, experiment_factory):
        """Parses a recommender experiment configuration.

        Args:
            experiment_config(dict): the experiment's total configuration.
            data_registry(DataRegistry): the data registry containing the available datasets.
            experiment_factory(GroupFactory): the factory containing all three pipeline factories.

        Returns:
            parsed_config(RecommenderExperimentConfig): the parsed configuration or None on failure.
        """
        if not self.parse_experiment_name(experiment_config):
            return None

        experiment_name = experiment_config[KEY_NAME]
        experiment_type = experiment_config[KEY_TYPE]

        experiment_datasets = self.parse_experiment_datasets(
            experiment_config,
            data_registry,
            experiment_factory.get_factory(KEY_DATASETS)
        )
        if experiment_datasets is None:
            return None

        experiment_models = self.parse_experiment_models(
            experiment_config,
            experiment_factory.get_factory(KEY_MODELS).get_factory(TYPE_RECOMMENDATION)
        )
        if experiment_models is None:
            return None

        experiment_evaluation = self.parse_experiment_evaluation(
            experiment_config,
            experiment_factory.get_factory(KEY_EVALUATION).get_factory(TYPE_RECOMMENDATION)
        )

        _, experiment_top_k = parse_config_param(
            experiment_config,
            'recommender experiment',
            ConfigValueParam(
                KEY_TOP_K,
                int,
                DEFAULT_TOP_K,
                (1, 100)
            ),
            self.event_dispatcher
        )

        _, experiment_rated_items_filter = parse_config_param(
            experiment_config,
            'recommender experiment',
            ConfigOptionParam(
                KEY_RATED_ITEMS_FILTER,
                bool,
                DEFAULT_RATED_ITEMS_FILTER,
                [True, False]
            ),
            self.event_dispatcher
        )

        return RecommenderExperimentConfig(
            experiment_datasets,
            experiment_models,
            experiment_evaluation,
            experiment_name,
            experiment_type,
            experiment_top_k,
            experiment_rated_items_filter
        )

    def parse_experiment_name(self, experiment_config):
        """Parses the name of the experiment.

        Args:
            experiment_config(dict): the experiment's total configuration.

        Returns:
            success(bool): whether the name is available in the configuration.
        """
        if not assert_is_key_in_dict(
            KEY_NAME,
            experiment_config,
            self.event_dispatcher,
            'PARSE ERROR: missing experiment key \'' + KEY_NAME + '\' (required)'
        ): return False

        if not assert_is_type(
            experiment_config[KEY_NAME],
            str,
            self.event_dispatcher,
            'PARSE ERROR: invalid value for experiment key \'' + KEY_NAME + '\''
        ): return False

        return True

    def parse_experiment_datasets(self, experiment_config, data_registry, data_factory):
        """Parses the datasets of the experiment.

        Args:
            experiment_config(dict): the experiment's total configuration.
            data_registry(DataRegistry): the data registry containing the available datasets.
            data_factory(GroupFactory):

        Returns:
            experiment_datasets(array like): list of parsed DatasetConfig's or None on failure.
        """
        experiment_datasets = parse_data_config(
            experiment_config,
            data_registry,
            data_factory,
            self.event_dispatcher
        )

        if not assert_is_container_not_empty(
            experiment_datasets,
            self.event_dispatcher,
            'PARSE ERROR: no experiment ' + KEY_DATASETS + ' specified'
        ): return None

        return experiment_datasets

    def parse_experiment_evaluation(self, experiment_config, metric_factory):
        """Parses the evaluation of the experiment.

        Args:
            experiment_config(dict): the experiment's total configuration.
            metric_factory(MetricFactory): the metric factory containing available metrics.

        Returns:
            experiment_evaluation(array like): list of parsed MetricConfig's or None on failure.
        """
        experiment_evaluation = parse_evaluation_config(
            experiment_config,
            metric_factory,
            self.event_dispatcher
        )

        return experiment_evaluation

    def parse_experiment_models(self, experiment_config, model_factory):
        """Parses the models of the experiment.

        Args:
            experiment_config(dict): the experiment's total configuration.
            model_factory(ModelFactory): the model factory containing the available models.

        Returns:
            experiment_models(dict): dictionary of parsed ModelConfig's keyed by API name
                or None on failure.
        """
        experiment_models = parse_models_config(
            experiment_config,
            model_factory,
            self.event_dispatcher
        )

        if not assert_is_container_not_empty(
            experiment_models,
            self.event_dispatcher,
            'PARSE ERROR: no experiment ' + KEY_MODELS + ' specified'
        ): return None

        return experiment_models

    def parse_experiment_top_k(self, recommender_experiment_config):
        """Parses the top K of the recommender experiment.

        Args:
            recommender_experiment_config(dict): the experiment's total configuration.

        Returns:
            top_k(int): the topK of the experiment or default_top_k on failure.
        """
        top_k = DEFAULT_TOP_K

        if recommender_experiment_config.get(KEY_TOP_K) is None:
            self.event_dispatcher.dispatch(
                ON_PARSE,
                msg='PARSE WARNING: missing experiment key \'' + KEY_TOP_K + '\'',
                default=top_k
            )
        else:
            top_k = recommender_experiment_config[KEY_TOP_K]

        return top_k

    def parse_experiment_type(self, experiment_config):
        """Parses the type of the experiment.

        Args:
            experiment_config(dict): the experiment's total configuration.

        Returns:
            experiment_type(str): the type of the experiment or None on failure.
        """
        # assert experiment_config is a dict
        if not assert_is_type(
            experiment_config,
            dict,
            self.event_dispatcher,
            'PARSE ERROR: invalid experiment config type'
        ): return None

        # assert KEY_TYPE is present
        if not assert_is_key_in_dict(
            KEY_TYPE,
            experiment_config,
            self.event_dispatcher,
            'PARSE ERROR: missing experiment key \'' + KEY_TYPE + '\' (required)',
            one_of_list=VALID_TYPES
        ): return None

        experiment_type = experiment_config[KEY_TYPE]

        # assert experiment_type is valid
        if not assert_is_one_of_list(
            experiment_type,
            VALID_TYPES,
            self.event_dispatcher,
            'PARSE ERROR: invalid value for experiment ' + KEY_TYPE +
            ' \'' + str(experiment_type) + '\''
        ): return None

        return experiment_type
