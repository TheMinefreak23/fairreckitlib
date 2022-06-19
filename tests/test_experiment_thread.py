"""This module tests the experiment threading functionality.

Functions:

    test_experiment_thread_failures: test the experiment thread to run into failures.
    test_experiment_thread_success: test the experiment thread to run successfully.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time

import pytest

from src.fairreckitlib.core.core_constants import VALID_TYPES
from src.fairreckitlib.core.threading.thread_processor import ThreadProcessor
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.experiment.experiment_event import ON_END_EXPERIMENT_THREAD
from src.fairreckitlib.experiment.experiment_factory import create_experiment_factory
from src.fairreckitlib.experiment.experiment_run import ExperimentPipelineConfig
from src.fairreckitlib.experiment.experiment_thread import ThreadExperiment
from .test_experiment_pipeline import create_experiment_config_duplicates
from .conftest import NUM_THREADS


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_experiment_thread_failures(
        experiment_type: str,
        data_registry: DataRegistry,
        io_tmp_dir: str) -> None:
    """Test the experiment thread to run into failures."""
    def assert_no_success(_, __, **kwargs) -> None:
        """Assert the experiment (result) to have failed."""
        assert not kwargs['success'], \
            'expected experiment to fail'

    experiment_factory = create_experiment_factory(data_registry)

    failure_configs = [
        # no data transitions
        create_experiment_config_duplicates(
            experiment_type,
            data_registry,
            experiment_factory,
            num_data_duplicates=0
        ),
        # no computed models
        create_experiment_config_duplicates(
            experiment_type,
            data_registry,
            experiment_factory,
            num_model_duplicates=0
        )
    ]

    for i, experiment_config in enumerate(failure_configs):
        pipeline_config = ExperimentPipelineConfig(
            os.path.join(io_tmp_dir, str(i)),
            data_registry,
            experiment_factory,
            experiment_config,
            0, 1,
            NUM_THREADS
        )

        experiment_thread = ThreadExperiment(
            'experiment',
            {ON_END_EXPERIMENT_THREAD: assert_no_success},
            True,
            **{'pipeline_config': pipeline_config}
        )

        thread_processor = ThreadProcessor()
        thread_processor.start(experiment_thread)
        while thread_processor.get_num_active() > 0:
            time.sleep(1)


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_experiment_thread_success(
        experiment_type: str,
        data_registry: DataRegistry,
        io_tmp_dir: str) -> None:
    """Test the experiment thread to run successfully."""
    def assert_success(_, __, **kwargs) -> None:
        """Assert the experiment (result) to have succeeded."""
        assert kwargs['success'], \
            'expected experiment to succeed'

    experiment_factory = create_experiment_factory(data_registry)

    pipeline_config = ExperimentPipelineConfig(
        io_tmp_dir,
        data_registry,
        experiment_factory,
        create_experiment_config_duplicates(
            experiment_type,
            data_registry,
            experiment_factory,
            num_data_duplicates=1,
            num_model_duplicates=1,
            num_metric_duplicates=1
        ),
        0, 1,
        NUM_THREADS
    )

    experiment_thread = ThreadExperiment(
        'experiment',
        {ON_END_EXPERIMENT_THREAD: assert_success},
        True,
        **{'pipeline_config': pipeline_config}
    )

    thread_processor = ThreadProcessor()
    thread_processor.start(experiment_thread)
    while thread_processor.get_num_active() > 0:
        time.sleep(1)
