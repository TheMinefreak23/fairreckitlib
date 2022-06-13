"""This module tests the formatting and parsing of the experiment configurations.

Functions:

    create_experiment_config: create experiment configuration with all availability.
    test_parse_experiment_config_failure: test parsing failure for both types.
    test_parse_experiment_config: test parsing total experiment configuration.
    test_parse_experiment_config_from_yml: test parsing total experiment configuration from yml.
    test_parse_experiment_name: test parsing the name from the config.
    test_parse_experiment_rated_items_filter: test parsing the rated items filter from the config.
    test_parse_experiment_top_k: test parsing the top k from the config.
    test_parse_experiment_type: test parsing the type from the config.
    assert_required_formatted_experiment_config: assert required entries to be present.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict

import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.core_constants import DEFAULT_TOP_K, KEY_TOP_K, MIN_TOP_K, MAX_TOP_K
from src.fairreckitlib.core.core_constants import DEFAULT_RATED_ITEMS_FILTER,KEY_RATED_ITEMS_FILTER
from src.fairreckitlib.core.core_constants import KEY_NAME, \
    KEY_TYPE, VALID_TYPES, TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.core.io.io_utility import save_yml
from src.fairreckitlib.data.data_factory import KEY_DATA
from src.fairreckitlib.data.set.dataset_constants import KEY_DATASET, KEY_MATRIX
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.evaluation.evaluation_factory import KEY_EVALUATION
from src.fairreckitlib.experiment.experiment_config import \
    ExperimentConfig, PredictorExperimentConfig, RecommenderExperimentConfig
from src.fairreckitlib.experiment.experiment_config_parser import ExperimentConfigParser
from src.fairreckitlib.experiment.experiment_factory import create_experiment_factory
from src.fairreckitlib.model.model_factory import KEY_MODELS

INVALID_CONTAINER_TYPES = [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]

parser = ExperimentConfigParser(verbose=True)


def create_experiment_config(
        dataset_registry: DataRegistry,
        experiment_factory: GroupFactory,
        experiment_type: str) -> Dict[str, Any]:
    """Create experiment configuration with the entire availability in the factory."""
    experiment_config = {
        KEY_NAME: 'experiment',
        KEY_TYPE: experiment_type,
        KEY_DATA: [],
        KEY_MODELS: {},
        KEY_EVALUATION: []
    }

    # these are mandatory for recommendation
    if experiment_type == TYPE_RECOMMENDATION:
        experiment_config[KEY_TOP_K] = DEFAULT_TOP_K
        experiment_config[KEY_RATED_ITEMS_FILTER] = DEFAULT_RATED_ITEMS_FILTER

    for dataset_name in dataset_registry.get_available_sets():
        dataset = dataset_registry.get_set(dataset_name)
        for matrix_name in dataset.get_available_matrices():
            experiment_config[KEY_DATA].append({
                KEY_DATASET: dataset_name,
                KEY_MATRIX: matrix_name
            })

    model_factory = experiment_factory.get_factory(KEY_MODELS).get_factory(experiment_type)
    for algo_api_name in model_factory.get_available_names():
        algo_api_factory = model_factory.get_factory(algo_api_name)
        experiment_config[KEY_MODELS][algo_api_name] = []

        for algo_name in algo_api_factory.get_available_names():
            experiment_config[KEY_MODELS][algo_api_name].append({
                KEY_NAME: algo_name
            })

    eval_factory = experiment_factory.get_factory(KEY_EVALUATION).get_factory(experiment_type)
    for metric_category_name in eval_factory.get_available_names():
        metric_category_factory = eval_factory.get_factory(metric_category_name)

        for metric_name in metric_category_factory.get_available_names():
            experiment_config[KEY_EVALUATION].append({
                KEY_NAME: metric_name
            })

    return experiment_config


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_parse_experiment_config_failure(
        data_registry: DataRegistry,
        experiment_type: str) -> None:
    """Test parsing failure for the total experiment configuration of both types."""
    experiment_factory = create_experiment_factory(data_registry)

    # test failure for parsing various types, including a dict that is empty
    for experiment_config in INVALID_CONTAINER_TYPES:
        parsed_experiment_config = parser.parse_experiment_config(
            experiment_config,
            data_registry,
            experiment_factory
        )
        assert not bool(parsed_experiment_config), \
            'expected failure when parsing an invalid configuration'

    experiment_config = create_experiment_config(
        data_registry,
        experiment_factory,
        experiment_type
    )

    # test failure when parsing configuration without an experiment type
    del experiment_config[KEY_TYPE]
    parsed_experiment_config = parser.parse_experiment_config(
        experiment_config,
        data_registry,
        experiment_factory
    )
    assert not bool(parsed_experiment_config), \
        'expected failure when missing experiment type'

    experiment_config = create_experiment_config(
        data_registry,
        experiment_factory,
        experiment_type
    )

    # test failure when parsing configuration without data matrices
    del experiment_config[KEY_DATA]
    parsed_experiment_config = parser.parse_experiment_config(
        experiment_config,
        data_registry,
        experiment_factory
    )
    assert not bool(parsed_experiment_config), \
        'expected failure when missing experiment datasets'

    experiment_config = create_experiment_config(
        data_registry,
        experiment_factory,
        experiment_type
    )

    # test failure when parsing configuration without models
    del experiment_config[KEY_MODELS]
    parsed_experiment_config = parser.parse_experiment_config(
        experiment_config,
        data_registry,
        experiment_factory
    )
    assert not bool(parsed_experiment_config), \
        'expected failure when missing experiment models'


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_parse_experiment_config(
        data_registry: DataRegistry,
        experiment_type: str) -> None:
    """Test parsing prediction and recommendation experiment configuration."""
    experiment_factory = create_experiment_factory(data_registry)

    experiment_config = create_experiment_config(
        data_registry,
        experiment_factory,
        experiment_type
    )

    # test success without evaluation configuration
    del experiment_config[KEY_EVALUATION]
    parsed_experiment_config = parser.parse_experiment_config(
        experiment_config,
        data_registry,
        experiment_factory
    )
    assert isinstance(parsed_experiment_config, ExperimentConfig), \
        'expected parsing to succeed because evaluation is optional'
    assert len(parsed_experiment_config.evaluation) == 0, \
        'did not expect the parsed evaluation to have metrics'

    formatted_experiment_config = parsed_experiment_config.to_yml_format()
    assert_required_formatted_experiment_config(formatted_experiment_config)
    assert KEY_EVALUATION not in formatted_experiment_config, \
        'did not expect evaluation metrics in the formatted experiment configuration'

    experiment_config = create_experiment_config(
        data_registry,
        experiment_factory,
        experiment_type
    )

    # test success with evaluation configuration
    parsed_experiment_config = parser.parse_experiment_config(
        experiment_config,
        data_registry,
        experiment_factory
    )
    assert isinstance(parsed_experiment_config, ExperimentConfig), \
        'expected parsing to succeed with evaluation metrics configuration'
    assert len(parsed_experiment_config.evaluation) > 0, \
        'expected the parsed evaluation configuration to have metrics'

    formatted_experiment_config = parsed_experiment_config.to_yml_format()
    assert_required_formatted_experiment_config(formatted_experiment_config)
    assert KEY_EVALUATION in formatted_experiment_config, \
        'expected evaluation metrics in the formatted experiment configuration'

    if experiment_type == TYPE_PREDICTION:
        assert isinstance(parsed_experiment_config, PredictorExperimentConfig), \
            'expected predictor experiment configuration to be parsed'
        assert KEY_TOP_K not in formatted_experiment_config, \
            'did not expect top k to be present for predictions'
        assert KEY_RATED_ITEMS_FILTER not in formatted_experiment_config, \
            'did not expect rated items filter to be present for predictions'
    elif experiment_type == TYPE_RECOMMENDATION:
        assert isinstance(parsed_experiment_config, RecommenderExperimentConfig), \
            'expected recommender experiment configuration to be parsed'
        assert KEY_TOP_K in formatted_experiment_config, \
            'expected top k to be present for recommendations'
        assert KEY_RATED_ITEMS_FILTER in formatted_experiment_config, \
            'expected rated items filter to be present for recommendations'
    else:
        raise TypeError('Unknown experiment type')


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_parse_experiment_config_from_yml(
        io_tmp_dir: str,
        data_registry: DataRegistry,
        experiment_type: str) -> None:
    """Test integration when parsing experiment configuration with yml."""
    experiment_factory = create_experiment_factory(data_registry)

    parsed_experiment_config = parser.parse_experiment_config(
        create_experiment_config(
            data_registry,
            experiment_factory,
            experiment_type
        ),
        data_registry,
        experiment_factory
    )
    formatted_experiment_config = parsed_experiment_config.to_yml_format()

    experiment_file = os.path.join(io_tmp_dir, 'experiment')
    save_yml(experiment_file + '.yml', formatted_experiment_config)
    parsed_experiment_config = parser.parse_experiment_config_from_yml(
        experiment_file,
        data_registry,
        experiment_factory
    )

    assert formatted_experiment_config == parsed_experiment_config.to_yml_format(), \
        'expected formatted configuration to be the same after reloading from yml'


def test_parse_experiment_name() -> None:
    """Test parsing the name from the experiment configuration."""
    assert not bool(parser.parse_experiment_name({})), \
        'expected failure when parsing an experiment configuration without a name'

    for experiment_name in [None, True, False, 0, 0.0, [], {}, {'set'}]:
        assert not bool(parser.parse_experiment_name({KEY_NAME: experiment_name})), \
            'expected failure when parsing an experiment configuration with an invalid name'

    experiment_name = 'experiment'
    assert parser.parse_experiment_name({KEY_NAME: experiment_name}) == experiment_name, \
        'expected success when parsing an experiment configuration with a name'


def test_parse_experiment_rated_items_filter() -> None:
    """Test parsing the rated items filter from the experiment configuration."""
    parsed_rated_items_filter = parser.parse_experiment_rated_items_filter({})
    assert parsed_rated_items_filter == DEFAULT_RATED_ITEMS_FILTER, \
        'expected default rated items filter when parsing a missing rated items filter'

    for rated_items_filter in [None, 0, 0.0, 'a', [], {}, {'set'}]:
        parsed_rated_items_filter = parser.parse_experiment_rated_items_filter({
            KEY_RATED_ITEMS_FILTER: rated_items_filter
        })
        assert parsed_rated_items_filter == DEFAULT_RATED_ITEMS_FILTER, \
            'expected default rated items filter when parsing an invalid rated items filter'


def test_parse_experiment_top_k() -> None:
    """Test parsing the top k from the experiment configuration."""
    parsed_top_k = parser.parse_experiment_top_k({})
    assert parsed_top_k == DEFAULT_TOP_K, \
        'expected default top k when parsing a missing top k'

    for top_k in [None, True, False, 'a', [], {}, {'set'}]:
        parsed_top_k = parser.parse_experiment_top_k({KEY_TOP_K: top_k})
        assert parsed_top_k == DEFAULT_TOP_K, \
            'expected default top k when parsing an invalid top k'

    for top_k in [MIN_TOP_K, DEFAULT_TOP_K, MAX_TOP_K]:
        parsed_top_k = parser.parse_experiment_top_k({KEY_TOP_K: top_k})
        assert parsed_top_k == top_k, \
            'expected parsing to succeed and to return the same top k as the input'

        parsed_top_k = parser.parse_experiment_top_k({KEY_TOP_K: float(top_k)})
        assert parsed_top_k == top_k, \
            'expected parsing to succeed and to return the casted top k'

    parsed_top_k = parser.parse_experiment_top_k({KEY_TOP_K: MIN_TOP_K - 1})
    assert parsed_top_k == MIN_TOP_K, \
        'expected top k to be clamped to min top k'

    parsed_top_k = parser.parse_experiment_top_k({KEY_TOP_K: MAX_TOP_K + 1})
    assert parsed_top_k == MAX_TOP_K, \
        'expected top k to be clamped to min top k'


def test_parse_experiment_type() -> None:
    """Test parsing the type from the experiment configuration."""
    assert not bool(parser.parse_experiment_type({})), \
        'expected failure when parsing an experiment configuration without a type'

    assert not bool(parser.parse_experiment_type({KEY_TYPE: 'unknown'})), \
        'expected failure when parsing an experiment configuration with an unknown type'

    for i, experiment_type in enumerate(VALID_TYPES):
        assert parser.parse_experiment_type({KEY_TYPE: experiment_type}) == VALID_TYPES[i], \
            'expected success when parsing an experiment configuration with a valid type'


def assert_required_formatted_experiment_config(
        formatted_experiment_config: Dict[str, Any]) -> None:
    """Assert the required entries to be present in the formatted experiment configuration."""
    assert KEY_NAME in formatted_experiment_config, \
        'expected name in the formatted experiment configuration'
    assert KEY_TYPE in formatted_experiment_config, \
        'expected type in the formatted experiment configuration'
    assert KEY_DATA in formatted_experiment_config, \
        'expected data matrices in the formatted experiment configuration'
    assert KEY_MODELS in formatted_experiment_config, \
        'expected model algorithms in the formatted experiment configuration'
