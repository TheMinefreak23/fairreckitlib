"""This module tests the experiment pipeline functionality.

Functions:

    test_run_experiment_pipelines_errors: test experiment pipeline (run) errors.
    test_run_experiment_pipelines: test the experiment pipeline (run) integration.
    test_resolve_experiment_start_run: test resolving the experiment result's start run.
    create_experiment_config_coverage: create experiment configuration that covers all.
    create_experiment_config_duplicates: create experiment configuration with duplicates.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Tuple, Union

import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.core_constants import DEFAULT_TOP_K, DEFAULT_RATED_ITEMS_FILTER, \
    VALID_TYPES, TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_dir, create_json
from src.fairreckitlib.core.io.io_utility import load_json
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.evaluation.evaluation_factory import KEY_EVALUATION
from src.fairreckitlib.experiment.experiment_config import \
    PredictorExperimentConfig, RecommenderExperimentConfig
from src.fairreckitlib.experiment.experiment_factory import create_experiment_factory
from src.fairreckitlib.experiment.experiment_pipeline import ExperimentPipeline
from src.fairreckitlib.experiment.experiment_run import \
    ExperimentPipelineConfig, run_experiment_pipelines, resolve_experiment_start_run
from src.fairreckitlib.model.model_factory import KEY_MODELS
from .conftest import NUM_THREADS, is_always_running
from .test_data_pipeline import create_data_matrix_config_list
from .test_model_pipeline import \
    create_model_type_config_coverage, create_model_type_config_duplicates
from .test_evaluation_pipeline import create_metric_config_list


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_run_experiment_pipelines_errors(
        experiment_type: str,
        io_tmp_dir: str,
        data_registry: DataRegistry,
        experiment_event_dispatcher: EventDispatcher) -> None:
    """Test experiment pipeline (run) errors."""
    experiment_pipeline = ExperimentPipeline(
        data_registry,
        create_experiment_factory(data_registry),
        experiment_event_dispatcher
    )

    data_list = create_data_matrix_config_list(data_registry, 1)

    experiment_config_list = []
    if experiment_type == TYPE_PREDICTION:
        experiment_config_list = [
            # no data transitions
            PredictorExperimentConfig([], {}, [], 'experiment'),
            # no computed models
            PredictorExperimentConfig(data_list, {}, [], 'experiment'),
        ]
    elif experiment_type == TYPE_RECOMMENDATION:
        experiment_config_list = [
            # no data transitions
            RecommenderExperimentConfig([], {}, [], 'experiment',
                                        DEFAULT_TOP_K, DEFAULT_RATED_ITEMS_FILTER),
            # no computed models
            RecommenderExperimentConfig(data_list, {}, [], 'experiment',
                                        DEFAULT_TOP_K, DEFAULT_RATED_ITEMS_FILTER)
        ]

    for i, experiment_config in enumerate(experiment_config_list):
        # test failure for already known experiment output directory
        pytest.raises(IOError, experiment_pipeline.run, io_tmp_dir,
                      experiment_config, NUM_THREADS, is_always_running)

        # test failure to generate data transitions / computed models
        output_dir = os.path.join(io_tmp_dir, str(i))
        pytest.raises(RuntimeError, experiment_pipeline.run, output_dir,
                      experiment_config, NUM_THREADS, is_always_running)
        # verify no overview json was created
        pytest.raises(FileNotFoundError, load_json, os.path.join(output_dir, 'overview.json'))

        # experiment run integration that catches runtime errors
        start_run = 0
        assert not run_experiment_pipelines(
            ExperimentPipelineConfig(
                output_dir,
                data_registry,
                create_experiment_factory(data_registry),
                experiment_config,
                start_run, 1,
                NUM_THREADS
            ),
            experiment_event_dispatcher,
            is_always_running
        ), 'expected the experiment pipeline to fail for no data transitions/computed models'
        # verify no overview json was created
        overview_json_path = os.path.join(io_tmp_dir, 'run_' + str(start_run), 'overview.json')
        pytest.raises(FileNotFoundError, load_json, overview_json_path)


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_run_experiment_pipelines(
        experiment_type: str,
        io_tmp_dir: str,
        data_registry: DataRegistry,
        experiment_event_dispatcher: EventDispatcher) -> None:
    """Test the experiment pipeline (run) integration."""
    start_run = 0
    experiment_factory = create_experiment_factory(data_registry)
    experiment_config, overview_total = create_experiment_config_coverage(
        experiment_type,
        data_registry,
        experiment_factory
    )

    assert run_experiment_pipelines(
        ExperimentPipelineConfig(
            io_tmp_dir,
            data_registry,
            experiment_factory,
            experiment_config,
            start_run, 1,
            NUM_THREADS
        ),
        experiment_event_dispatcher,
        is_always_running
    ), 'expected the experiment pipeline to succeed'

    run_dir = os.path.join(io_tmp_dir, 'run_' + str(start_run))
    assert os.path.isdir(run_dir), \
        'expected run directory to be present'

    # directory contains the train and test set as well
    data_dirs = os.listdir(run_dir)
    num_data_dirs = len([d for d in data_dirs if os.path.isdir(os.path.join(run_dir, d))])
    assert num_data_dirs == len(experiment_config.datasets), \
        'expected output directory for each dataset'

    overview_json_path = os.path.join(run_dir, 'overview.json')
    assert os.path.isfile(overview_json_path), \
        'expected overview json to be created'

    overview_json = load_json(overview_json_path)
    assert len(overview_json['overview']) == overview_total, \
        'expected all data-model pairs to be present in the overview result'


def test_resolve_experiment_start_run(io_tmp_dir, io_event_dispatcher: EventDispatcher) -> None:
    """Test resolving the experiment start run in (result) directory."""
    # test failure for unknown result directory
    pytest.raises(IOError, resolve_experiment_start_run, os.path.join(io_tmp_dir, 'unknown'))

    assert resolve_experiment_start_run(io_tmp_dir) == 0, \
        'expected start run to be 0 for no present run directories'

    create_json(os.path.join(io_tmp_dir, 'overview.json'), {}, io_event_dispatcher)
    assert resolve_experiment_start_run(io_tmp_dir) == 0, \
        'expected start run to be 0 when skipping existing files'

    create_dir(os.path.join(io_tmp_dir, 'unknown'), io_event_dispatcher)
    assert resolve_experiment_start_run(io_tmp_dir) == 0, \
        'expected start run to be 0 when skipping unknown run directories'

    create_dir(os.path.join(io_tmp_dir, 'run_a'), io_event_dispatcher)
    assert resolve_experiment_start_run(io_tmp_dir) == 0, \
        'expected start run to be 0 when the run directory does not have a number'

    for i in range(5):
        create_dir(os.path.join(io_tmp_dir, 'run_' + str(i)), io_event_dispatcher)
        assert resolve_experiment_start_run(io_tmp_dir) == i + 1, \
            'expected start run to be resolved'


def create_experiment_config_coverage(
        experiment_type: str,
        datasets_registry: DataRegistry,
        experiment_factory: GroupFactory
    ) -> Tuple[Union[PredictorExperimentConfig, RecommenderExperimentConfig], int]:
    """Create experiment configuration that covers all datasets/model(APIs)/metrics."""
    data_list = create_data_matrix_config_list(datasets_registry, 1)
    model_dict, num_models = create_model_type_config_coverage(
        experiment_factory.get_factory(KEY_MODELS).get_factory(experiment_type)
    )
    metric_list = create_metric_config_list(
        experiment_factory.get_factory(KEY_EVALUATION).get_factory(experiment_type),
        1
    )
    overview_total = len(data_list) * num_models

    if experiment_type == TYPE_PREDICTION:
        return PredictorExperimentConfig(
            data_list, model_dict, metric_list, 'experiment'
        ), overview_total
    if experiment_type == TYPE_RECOMMENDATION:
        return RecommenderExperimentConfig(
            data_list, model_dict, metric_list, 'experiment',
            DEFAULT_TOP_K, DEFAULT_RATED_ITEMS_FILTER
        ), overview_total

    raise TypeError('Unknown experiment type')


def create_experiment_config_duplicates(
        experiment_type: str,
        datasets_registry: DataRegistry,
        experiment_factory: GroupFactory,
        *,
        num_data_duplicates: int=1,
        num_model_duplicates: int=1,
        num_metric_duplicates: int=1,
    ) -> Union[PredictorExperimentConfig, RecommenderExperimentConfig]:
    """Create experiment configuration with duplicate datasets/models/metrics."""
    data_list = create_data_matrix_config_list(datasets_registry, num_data_duplicates)

    model_dict = create_model_type_config_duplicates(
        experiment_factory.get_factory(KEY_MODELS).get_factory(experiment_type),
        num_model_duplicates
    )
    metric_list = create_metric_config_list(
        experiment_factory.get_factory(KEY_EVALUATION).get_factory(experiment_type),
        num_metric_duplicates
    )

    if experiment_type == TYPE_PREDICTION:
        return PredictorExperimentConfig(
            data_list, model_dict, metric_list, 'experiment'
        )
    if experiment_type == TYPE_RECOMMENDATION:
        return RecommenderExperimentConfig(
            data_list, model_dict, metric_list, 'experiment',
            DEFAULT_TOP_K, DEFAULT_RATED_ITEMS_FILTER
        )

    raise TypeError('Unknown experiment type')
