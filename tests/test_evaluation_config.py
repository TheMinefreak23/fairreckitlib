"""This module tests the formatting and parsing of the evaluation configuration of the experiment.

Functions:

    create_metrics_config: create metrics config list with all available metrics.
    test_parse_missing_evaluation: test parsing missing evaluation from the experiment config.
    test_parse_evaluation_config: test parsing metrics from the evaluation configuration.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List

import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.config.config_yml import format_yml_config_list
from src.fairreckitlib.core.core_constants import \
    KEY_NAME, KEY_PARAMS, KEY_TOP_K, DEFAULT_TOP_K, TYPE_RECOMMENDATION, VALID_TYPES
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.data.filter.filter_config import DataSubsetConfig
from src.fairreckitlib.data.filter.filter_constants import KEY_DATA_FILTER_PASS, KEY_DATA_SUBSET
from src.fairreckitlib.data.filter.filter_factory import create_filter_factory
from src.fairreckitlib.data.set.dataset_constants import KEY_DATASET, KEY_MATRIX
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.evaluation.evaluation_factory import \
    KEY_EVALUATION, create_evaluation_factory
from src.fairreckitlib.evaluation.metrics.metric_factory import KEY_METRIC_PARAM_K
from src.fairreckitlib.evaluation.pipeline.evaluation_config import KEY_METRIC_SUBGROUP,MetricConfig
from src.fairreckitlib.evaluation.pipeline.evaluation_config_parsing import parse_evaluation_config


evaluation_factory = create_evaluation_factory()

INVALID_CONTAINER_TYPES = \
    [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]


def create_metrics_config(
        eval_type_factory: GroupFactory,
        data_filter_factory: GroupFactory=None) -> List[Dict[str, Any]]:
    """Create metrics config list with all available metrics of the evaluation type."""
    metrics_config = []
    for category_name in eval_type_factory.get_available_names():
        category_factory = eval_type_factory.get_factory(category_name)
        for metric_name in category_factory.get_available_names():
            # create metric configuration without subgroup
            if data_filter_factory is None:
                metrics_config.append({
                    KEY_NAME: metric_name,
                    KEY_PARAMS: category_factory.create_params(metric_name).get_defaults()
                })
            # create metric configuration with subgroup for each dataset-matrix pair
            else:
                for dataset_name in data_filter_factory.get_available_names():
                    dataset_filter_factory = data_filter_factory.get_factory(dataset_name)

                    for matrix_name in dataset_filter_factory.get_available_names():
                        matrix_filter_factory = dataset_filter_factory.get_factory(matrix_name)

                        subset = [{
                            KEY_DATA_FILTER_PASS: [{KEY_NAME: name}]
                        } for name in matrix_filter_factory.get_available_names()]

                        metrics_config.append({
                            KEY_NAME: metric_name,
                            KEY_PARAMS: category_factory.create_params(metric_name).get_defaults(),
                            KEY_METRIC_SUBGROUP: {
                                KEY_DATASET: dataset_name,
                                KEY_MATRIX: matrix_name,
                                KEY_DATA_SUBSET: subset
                            }
                        })

    return metrics_config


@pytest.mark.parametrize('eval_type', VALID_TYPES)
def test_parse_missing_evaluation(eval_type: str, parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing the evaluation configuration from the experiment configuration."""
    eval_type_factory = evaluation_factory.get_factory(eval_type)
    data_registry = None # not used
    data_filter_factory = GroupFactory('not used')

    parsed_metrics = parse_evaluation_config(
        data_registry,
        data_filter_factory,
        {},
        eval_type_factory,
        parse_event_dispatcher
    )
    assert not parsed_metrics, \
        'did not expect any parsed metrics for invalid evaluation configuration'

    # test failure for parsing various types, including a list that is empty
    for metric_config_list in INVALID_CONTAINER_TYPES:
        parsed_metrics = parse_evaluation_config(
            data_registry,
            data_filter_factory,
            {KEY_EVALUATION: metric_config_list},
            eval_type_factory,
            parse_event_dispatcher
        )

        assert not parsed_metrics, \
            'did not expect any parsed metrics for invalid evaluation configuration'


@pytest.mark.parametrize('eval_type', VALID_TYPES)
def test_parse_evaluation_config(
        eval_type: str,
        data_registry: DataRegistry,
        parse_event_dispatcher: EventDispatcher) -> None:
    """Test parsing metrics from the evaluation configuration for both evaluation types."""
    data_filter_factory = None # not used

    eval_type_factory = evaluation_factory.get_factory(eval_type)
    metric_config_list = create_metrics_config(eval_type_factory)
    experiment_config = {KEY_EVALUATION: metric_config_list}
    if eval_type == TYPE_RECOMMENDATION:
        experiment_config[KEY_TOP_K] = DEFAULT_TOP_K

    # test parsing success for the entire configuration, including the formatted parse result
    parsed_metrics = parse_evaluation_config(
        data_registry,
        data_filter_factory,
        experiment_config,
        eval_type_factory,
        parse_event_dispatcher
    )
    assert isinstance(parsed_metrics, list), \
        'expected list of metric configurations to be parsed'
    formatted_metrics_config = format_yml_config_list(parsed_metrics)

    assert len(formatted_metrics_config) == len(metric_config_list), \
        'expected all api algorithms to be present in the formatted metrics configuration'
    assert len(parsed_metrics) == len(metric_config_list), \
        'expected all metrics to be parsed from the original metric configuration list'

    for parsed_metric in parsed_metrics:
        assert isinstance(parsed_metric, MetricConfig), \
            'expected parsed metric to be a MetricConfig'
        assert not bool(parsed_metric.subgroup), \
            'did not expect metric subgroup to be parsed'
        assert any(parsed_metric.name is metric[KEY_NAME] for metric in formatted_metrics_config), \
            'expected parsed metric to be present in the formatted metrics configuration'
        assert any(parsed_metric.name is metric[KEY_NAME] for metric in metric_config_list), \
            'expected parsed metric to be present in the original metric configuration list'

        if KEY_METRIC_PARAM_K in parsed_metric.params:
            assert parsed_metric.params[KEY_METRIC_PARAM_K] == experiment_config[KEY_TOP_K], \
                'expected experiment top k to be used as default k for the accuracy metric'

    # parsing integration with metric subgroups
    data_filter_factory = create_filter_factory(data_registry)
    metric_config_list = create_metrics_config(eval_type_factory, data_filter_factory)
    experiment_config = {KEY_EVALUATION: metric_config_list}
    if eval_type == TYPE_RECOMMENDATION:
        experiment_config[KEY_TOP_K] = DEFAULT_TOP_K

    # test parsing success with subgroup
    parsed_metrics = parse_evaluation_config(
        data_registry,
        data_filter_factory,
        experiment_config,
        eval_type_factory,
        parse_event_dispatcher
    )
    for parsed_metric in parsed_metrics:
        assert isinstance(parsed_metric, MetricConfig), \
            'expected parsed metric to be a MetricConfig'
        assert bool(parsed_metric.subgroup), \
            'expected metric subgroup to be parsed'
        assert isinstance(parsed_metric.subgroup, DataSubsetConfig), \
            'expected DataSubsetConfig to be parsed as metric subgroup'
