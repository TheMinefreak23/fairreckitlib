"""This module tests the evaluation pipeline functionality.

Functions:

    test_evaluation_pipeline_set_errors: test evaluation pipeline for set loading errors.
    test_evaluation_pipeline_metric_errors: test evaluation pipeline for creation/evaluate errors.
    test_evaluation_pipeline_early_stop: test the early stopping of the evaluation pipeline.
    test_run_evaluation_pipelines:
    create_metric_config_list: create configuration list for all available metrics.
    assert_metric_evaluation_json: assert json with the evaluations of metrics present.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import List

import pandas as pd
import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.core_constants import KEY_NAME, KEY_PARAMS, VALID_TYPES
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_delete import delete_file
from src.fairreckitlib.core.io.io_utility import load_json
from src.fairreckitlib.data.data_transition import DataTransition
from src.fairreckitlib.data.filter.filter_factory import create_filter_factory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.evaluation.evaluation_factory import create_evaluation_factory
from src.fairreckitlib.evaluation.evaluation_sets import EvaluationSetPaths
from src.fairreckitlib.evaluation.metrics.metric_constants import \
    KEY_METRIC_EVALUATION, KEY_METRIC_SUBGROUP
from src.fairreckitlib.evaluation.pipeline.evaluation_config import MetricConfig
from src.fairreckitlib.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline
from src.fairreckitlib.evaluation.pipeline.evaluation_run import \
    EvaluationPipelineConfig, run_evaluation_pipelines
from .conftest import is_always_running
from .test_evaluation_metrics import DummyMetric


def test_evaluation_pipeline_set_errors(
        io_tmp_dir: str,
        eval_event_dispatcher: EventDispatcher) -> None:
    """Test evaluation pipeline for errors that can be raised during set loading."""
    dummy_set_path = os.path.join(io_tmp_dir, 'dummy_set.tsv')
    pd.DataFrame().to_csv(dummy_set_path, sep='\t')

    eval_pipeline = EvaluationPipeline(
        None, # not used
        None, # not used
        None, # not used
        eval_event_dispatcher
    )

    set_paths = EvaluationSetPaths('unknown', 'unknown', 'unknown')
    # test failure for unknown ratings set path
    pytest.raises(FileNotFoundError, eval_pipeline.load_evaluation_sets, set_paths, False, False)

    set_paths.ratings_path = dummy_set_path
    # test failure for unknown train set path
    pytest.raises(FileNotFoundError, eval_pipeline.load_evaluation_sets, set_paths, True, False)

    set_paths.train_path = dummy_set_path
    # test failure for unknown test set path
    pytest.raises(FileNotFoundError, eval_pipeline.load_evaluation_sets, set_paths, True, True)

    set_paths.test_path = dummy_set_path
    # test success for train and test set required
    sets = eval_pipeline.load_evaluation_sets(set_paths, True, True)
    assert sets.train is not None and sets.test is not None

    # test success for test set required
    set_paths.train_path = None
    set_paths.test_path = dummy_set_path
    sets = eval_pipeline.load_evaluation_sets(set_paths, False, True)
    assert sets.train is None and sets.test is not None

    # test success for train set required
    set_paths.train_path = dummy_set_path
    set_paths.test_path = None
    sets = eval_pipeline.load_evaluation_sets(set_paths, True, False)
    assert sets.train is not None and sets.test is None


@pytest.mark.parametrize('eval_type', VALID_TYPES)
def test_evaluation_pipeline_metric_errors(
        io_tmp_dir: str,
        eval_type: str,
        eval_event_dispatcher: EventDispatcher) -> None:
    """Test evaluation pipeline for errors that can be raised during metric creation/evaluation."""
    eval_type_factory = create_evaluation_factory().get_factory(eval_type)

    dummy_set_path = os.path.join(io_tmp_dir, 'dummy_set.tsv')
    pd.DataFrame().to_csv(dummy_set_path, sep='\t')

    for metric_category_name in eval_type_factory.get_available_names():
        metric_category_factory = eval_type_factory.get_factory(metric_category_name)\

        eval_pipeline = EvaluationPipeline(
            None, # not used,
            None, # not used,
            metric_category_factory,
            eval_event_dispatcher
        )

        # test failure for unknown metric
        print('\nEvaluationPipeline: construction warning')
        warning_json_path = os.path.join(
            io_tmp_dir, eval_type + '_' + metric_category_name + '_CreateWarning' + '.json'
        )
        eval_pipeline.run(
            warning_json_path,
            EvaluationSetPaths('', '', ''), # not used
            [MetricConfig('unknown', {}, None)],
            is_always_running
        )
        assert_metric_evaluation_json(warning_json_path, [])

        eval_error_tuples = [
            (ArithmeticError, 'ArithmeticError'),
            (MemoryError, 'MemoryError'),
            (RuntimeError, 'RuntimeError'),
        ]

        # test errors for construction and evaluating metrics
        for error, error_name in eval_error_tuples:
            metric_category_factory.add_obj(error_name, DummyMetric, None)
            metric_config = MetricConfig(error_name, {}, None)
            error_json_path = os.path.join(
                io_tmp_dir,
                eval_type + '_' + metric_category_name + '_' + error_name + '.json'
            )

            print('\nEvaluationPipeline: construction error =>', error_name)
            metric_kwargs = {
                'requires_test_set': False,
                'requires_train_set': False,
                'const_error': error
            }
            eval_pipeline.run(error_json_path, EvaluationSetPaths('', '', ''),
                              [metric_config], is_always_running, **metric_kwargs)
            assert_metric_evaluation_json(error_json_path, [])

            print('\nEvaluationPipeline: evaluation error =>', error_name)
            metric_kwargs = {
                'requires_test_set': False,
                'requires_train_set': False,
                'eval_error': error
            }
            eval_pipeline.run(error_json_path, EvaluationSetPaths(dummy_set_path, '', ''),
                              [metric_config], is_always_running, **metric_kwargs)
            assert_metric_evaluation_json(error_json_path, [])


@pytest.mark.parametrize('_eval_type', VALID_TYPES)
def test_evaluation_pipeline_early_stop(_eval_type: str) -> None:
    """Test the early stopping return statements of the evaluation pipeline."""
    # TODO test evaluation pipeline early stopping


@pytest.mark.parametrize('eval_type', VALID_TYPES)
def test_run_evaluation_pipelines(
        data_registry: DataRegistry,
        eval_type: str,
        eval_event_dispatcher: EventDispatcher) -> None:
    """Test the evaluation pipeline (run) integration."""
    eval_type_factory = create_evaluation_factory().get_factory(eval_type)
    data_filter_factory = create_filter_factory(data_registry)

    eval_set_dir = os.path.join('tests', 'evaluation_sets', eval_type)

    for data_dir in os.listdir(eval_set_dir):
        if not os.path.isdir(os.path.join(eval_set_dir, data_dir)):
            continue

        data_split = data_dir.split('_')
        data_dir = os.path.join(eval_set_dir, data_dir)

        model_dirs = []
        for model_dir in os.listdir(data_dir):
            model_dir = os.path.join(data_dir, model_dir)
            if not os.path.isdir(model_dir):
                continue

            model_dirs.append(model_dir)

        metric_config_list = create_metric_config_list(eval_type_factory)

        pipeline_config = EvaluationPipelineConfig(
            model_dirs,
            DataTransition(
                data_registry.get_set(data_split[0]), # dataset (name)
                data_split[1], # matrix name
                '',  # not used
                os.path.join(data_dir, 'train_set.tsv'),
                os.path.join(data_dir, 'test_set.tsv'),
                (0.0, 0.0)  # not used
            ),
            data_filter_factory,
            eval_type_factory,
            metric_config_list
        )

        run_evaluation_pipelines(pipeline_config, eval_event_dispatcher, is_always_running)

        for model_dir in os.listdir(data_dir):
            model_dir = os.path.join(data_dir, model_dir)
            if not os.path.isdir(model_dir):
                continue

            eval_json_path = os.path.join(model_dir, 'evaluations.json')

            assert_metric_evaluation_json(eval_json_path, metric_config_list)

            delete_file(eval_json_path, eval_event_dispatcher)


def create_metric_config_list(eval_type_factory: GroupFactory) -> List[MetricConfig]:
    """Create a metric configuration for all available metrics in the factory."""
    metric_config_list = []
    for metric_category_name in eval_type_factory.get_available_names():
        metric_category_factory = eval_type_factory.get_factory(metric_category_name)

        for metric_name in metric_category_factory.get_available_names():
            metric_config_list.append(MetricConfig(
                metric_name,
                metric_category_factory.create_params(metric_name).get_defaults(),
                None
            ))

    return metric_config_list


def assert_metric_evaluation_json(
        json_file_path: str, metric_config_list: List[MetricConfig]) -> None:
    """Assert the json to be created, with the evaluations of the specified metrics present."""
    assert os.path.isfile(json_file_path), \
        'expected evaluations json to be created'

    eval_json = load_json(json_file_path)
    assert len(eval_json['evaluations']) == len(metric_config_list), \
        'expected evaluation for each metric to be present'

    for metric_entry in eval_json['evaluations']:
        assert KEY_NAME in metric_entry, \
            'expected name of the metric in the evaluation'
        assert any(metric_entry[KEY_NAME] == m.name for m in metric_config_list), \
            'expected metric to be present in the original metric configuration list'
        assert KEY_PARAMS in metric_entry, \
            'expected params of the metric in the evaluation'

        assert KEY_METRIC_EVALUATION in metric_entry, \
            'expected evaluation to be present for the metric'

        metric_eval = metric_entry[KEY_METRIC_EVALUATION]

        assert 'value' in metric_eval, \
            'expected metric evaluation value to be present'
        assert isinstance(metric_eval['value'], float), \
            'expected metric evaluation value to be a floating-point value'
        assert KEY_METRIC_SUBGROUP in metric_eval, \
            'expected metric evaluation subgroup to be present'
        assert len(metric_eval[KEY_METRIC_SUBGROUP]) == 0, \
            'expected metric evaluation subgroup to be empty'
