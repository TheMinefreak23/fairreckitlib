"""This module contains a parser for the experiment configuration.

Classes:

    Parser: parse an experiment configuration from a dictionary or yml.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Optional, Union

from ..core.config_constants import KEY_TYPE, TYPE_PREDICTION, TYPE_RECOMMENDATION, VALID_TYPES
from ..core.config_constants import KEY_NAME, KEY_TOP_K, DEFAULT_TOP_K
from ..core.config_constants import KEY_RATED_ITEMS_FILTER, DEFAULT_RATED_ITEMS_FILTER
from ..core.config_params import ConfigOptionParam, ConfigValueParam
from ..core.event_dispatcher import EventDispatcher
from ..core.factories import GroupFactory
from ..core.parsing.parse_assert import assert_is_type
from ..core.parsing.parse_assert import assert_is_key_in_dict, assert_is_one_of_list
from ..core.parsing.parse_event import ON_PARSE, on_parse
from ..core.parsing.parse_params import parse_config_param
from ..data.data_factory import KEY_DATA
from ..data.pipeline.data_config_parsing import parse_data_config
from ..data.set.dataset_registry import DataRegistry
from ..data.utility import load_yml
from ..evaluation.evaluation_factory import KEY_EVALUATION
from ..evaluation.pipeline.evaluation_config_parsing import parse_evaluation_config
from ..model.model_factory import KEY_MODELS
from ..model.pipeline.model_config_parsing import parse_models_config
from .experiment_config import PredictorExperimentConfig, RecommenderExperimentConfig


class Parser:
    """Experiment Configuration Parser.

    Public methods:

    parse_experiment_config
    parse_experiment_config_from_yml
    """

    def __init__(self, verbose: bool):
        """Construct the Parser.

        Args:
            verbose: whether the parser should give verbose output.
        """
        self.verbose = verbose
        self.event_dispatcher = EventDispatcher()
        self.event_dispatcher.add_listener(ON_PARSE, self, (on_parse, None))

    def parse_experiment_config(
            self,
            experiment_config: Dict[str, Any],
            data_registry: DataRegistry,
            experiment_factory: GroupFactory
        ) -> Optional[Union[PredictorExperimentConfig, RecommenderExperimentConfig]]:
        """Parse an experiment configuration.

        Args:
            experiment_config: the experiment's total configuration.
            data_registry: the data registry containing the available datasets.
            experiment_factory: the factory containing all three pipeline factories.

        Returns:
            the parsed configuration or None on failure.
        """
        # assert experiment_config is a dict
        if not assert_is_type(
            experiment_config,
            dict,
            self.event_dispatcher,
            'PARSE ERROR: invalid experiment config type'
        ): return None

        # attempt to parse experiment name
        experiment_name = self.parse_experiment_name(experiment_config)
        if experiment_name is None:
            return None

        # attempt to parse experiment type
        experiment_type = self.parse_experiment_type(experiment_config)
        if experiment_type is None:
            return None

        # attempt to parse experiment datasets
        experiment_datasets = parse_data_config(
            experiment_config,
            data_registry,
            experiment_factory.get_factory(KEY_DATA),
            self.event_dispatcher
        )
        # experiment requires datasets
        if experiment_datasets is None:
            return None

        # attempt to parse experiment models
        experiment_models = parse_models_config(
            experiment_config,
            experiment_factory.get_factory(KEY_MODELS).get_factory(experiment_type),
            self.event_dispatcher
        )
        # experiment requires models
        if experiment_models is None:
            return None

        # attempt to parse experiment evaluation (optional)
        experiment_evaluation = parse_evaluation_config(
            experiment_config,
            experiment_factory.get_factory(KEY_EVALUATION).get_factory(experiment_type),
            self.event_dispatcher
        )

        parsed_config = None

        if experiment_type == TYPE_PREDICTION:
            parsed_config = PredictorExperimentConfig(
                experiment_datasets,
                experiment_models,
                experiment_evaluation,
                experiment_name,
                experiment_type
            )
        elif experiment_type == TYPE_RECOMMENDATION:
            # parse top k
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

            # parse rated items filter
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

            parsed_config = RecommenderExperimentConfig(
                experiment_datasets,
                experiment_models,
                experiment_evaluation,
                experiment_name,
                experiment_type,
                experiment_top_k,
                experiment_rated_items_filter
            )

        return parsed_config

    def parse_experiment_config_from_yml(
            self,
            file_path: str,
            data_registry: DataRegistry,
            experiment_factory: GroupFactory
        ) -> Optional[Union[PredictorExperimentConfig, RecommenderExperimentConfig]]:
        """Parse an experiment configuration from a yml file.

        Args:
            file_path: path to the yml file without extension.
            data_registry: the data registry containing the available datasets.
            experiment_factory: the factory containing all three pipeline factories.

        Returns:
            the parsed configuration or None on failure.
        """
        experiment_config = load_yml(file_path + '.yml')

        return self.parse_experiment_config(experiment_config, data_registry, experiment_factory)

    def parse_experiment_name(self, experiment_config: Dict[str, Any]) -> Optional[str]:
        """Parse the name of the experiment.

        Args:
            experiment_config(dict): the experiment's total configuration.

        Returns:
            the name of the experiment or None on failure.
        """
        if not assert_is_key_in_dict(
            KEY_NAME,
            experiment_config,
            self.event_dispatcher,
            'PARSE ERROR: missing experiment key \'' + KEY_NAME + '\' (required)'
        ): return None

        if not assert_is_type(
            experiment_config[KEY_NAME],
            str,
            self.event_dispatcher,
            'PARSE ERROR: invalid value for experiment key \'' + KEY_NAME + '\''
        ): return None

        return experiment_config[KEY_NAME]

    def parse_experiment_top_k(self, recommender_experiment_config: Dict[str, Any]) -> int:
        """Parse the top K of the recommender experiment.

        Args:
            recommender_experiment_config: the experiment's total configuration.

        Returns:
            the topK of the experiment or default_top_k on failure.
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

    def parse_experiment_type(self, experiment_config: Dict[str, Any]) -> Optional[str]:
        """Parse the type of the experiment.

        Args:
            experiment_config: the experiment's total configuration.

        Returns:
            the type of the experiment or None on failure.
        """
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
