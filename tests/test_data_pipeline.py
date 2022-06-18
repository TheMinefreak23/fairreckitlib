"""This module tests the data pipeline functionality.

Functions:

    test_data_pipeline_io_errors: test data pipeline for various IO errors.
    test_data_pipeline_runtime_errors: test data pipeline for runtime errors.
    test_data_pipeline_early_stop: test the early stopping of the data pipeline.
    test_run_data_pipelines_failures: test data pipeline run failure for warnings and errors.
    test_run_data_pipelines: test the data pipeline (run) integration.
    create_data_matrix_config_list: create data matrix configuration list for all datasets.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import List

import pytest

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.data.data_factory import create_data_factory
from src.fairreckitlib.data.pipeline.data_config import DataMatrixConfig
from src.fairreckitlib.data.pipeline.data_pipeline import DataPipeline
from src.fairreckitlib.data.pipeline.data_run import DataPipelineConfig, run_data_pipelines
from src.fairreckitlib.data.ratings.convert_config import ConvertConfig
from src.fairreckitlib.data.ratings.convert_constants import CONVERTER_RANGE, RATING_TYPE_THRESHOLD
from src.fairreckitlib.data.set.dataset_config import DatasetFileConfig, FileOptionsConfig
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.data.split.split_config import SplitConfig, create_default_split_config
from .conftest import is_always_running
from .test_dataset import create_dataset_with_dummy_matrix


def test_data_pipeline_io_errors(
        io_tmp_dir: str,
        data_event_dispatcher: EventDispatcher) -> None:
    """Test data pipeline for various IO errors that can be raised."""
    data_pipeline = DataPipeline(
        None, # not used
        data_event_dispatcher
    )

    # test run failure for unknown output directory
    pytest.raises(
        IOError,
        data_pipeline.run,
        os.path.join(io_tmp_dir, 'unknown'),
        None, # not used
        None, # not used
        is_always_running
    )

    matrix_name = 'matrix with unknown file'
    dataset = create_dataset_with_dummy_matrix(
        io_tmp_dir,
        'dataset',
        matrix_name,
        DatasetFileConfig('unknown.tsv', FileOptionsConfig(None, None, None, False, False))
    )

    # test load from dataset failure
    pytest.raises(FileNotFoundError, data_pipeline.load_from_dataset, dataset, matrix_name)
    # integration test load from dataset failure
    pytest.raises(
        FileNotFoundError,
        data_pipeline.run,
        io_tmp_dir,
        dataset,
        DataMatrixConfig(
            dataset.get_name(),
            matrix_name,
            [],
            None,
            create_default_split_config()
        ),
        is_always_running
    )


def test_data_pipeline_runtime_errors(
        io_tmp_dir: str,
        data_registry: DataRegistry,
        data_event_dispatcher: EventDispatcher) -> None:
    """Test data pipeline for runtime errors that can be raised."""
    data_pipeline = DataPipeline(
        create_data_factory(data_registry),
        data_event_dispatcher
    )

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            # test failure for unknown rating converter
            data_matrix_config = DataMatrixConfig(
                dataset.get_name(),
                matrix_name,
                [],
                ConvertConfig('unknown', {}),
                create_default_split_config()
            )

            pytest.raises(
                RuntimeError,
                data_pipeline.run,
                io_tmp_dir,
                dataset,
                data_matrix_config,
                is_always_running
            )
            assert data_pipeline.split_datasets[data_matrix_config.get_data_matrix_name()] == 0, \
                'did not expect the data matrix counter to be incremented'

            # test failure for unknown splitter
            data_matrix_config.converter = None
            data_matrix_config.splitting = SplitConfig('unknown', {}, 0.2)
            pytest.raises(
                RuntimeError,
                data_pipeline.run,
                io_tmp_dir,
                dataset,
                data_matrix_config,
                is_always_running
            )
            assert data_pipeline.split_datasets[data_matrix_config.get_data_matrix_name()] == 0, \
                'did not expect the data matrix counter to be incremented'


def test_data_pipeline_early_stop() -> None:
    """Test the early stopping return statements of the data pipeline."""
    # TODO data model pipeline early stopping


def test_run_data_pipelines_failures(
        io_tmp_dir: str,
        data_registry: DataRegistry,
        data_event_dispatcher: EventDispatcher) -> None:
    """Test the data pipeline run failure for warnings and errors."""
    # test empty configuration list
    pipeline_config = DataPipelineConfig(
        io_tmp_dir,
        data_registry,
        create_data_factory(data_registry),
        []
    )

    data_transitions = run_data_pipelines(
        pipeline_config,
        data_event_dispatcher,
        is_always_running
    )
    assert len(data_transitions) == 0, \
        'did not expect data transitions for an empty configuration list'

    # test skipping of configurations with an unknown dataset
    pipeline_config.data_config = [DataMatrixConfig(
        'unknown',
        '',
        [],
        None,
        create_default_split_config()
    )]

    data_transitions = run_data_pipelines(
        pipeline_config,
        data_event_dispatcher,
        is_always_running
    )
    assert len(data_transitions) == 0, \
        'did not expect data transitions for a configuration with an unknown dataset'

    # test skipping of configurations with an unknown dataset matrix
    for dataset_name in data_registry.get_available_sets():
        pipeline_config.data_config = [DataMatrixConfig(
            dataset_name,
            'unknown',
            [],
            None,
            create_default_split_config()
        )]

        data_transitions = run_data_pipelines(
            pipeline_config,
            data_event_dispatcher,
            is_always_running
        )
        assert len(data_transitions) == 0, \
            'did not expect data transitions for a configuration with an unknown matrix'

        # test skipping of configurations that raise a RuntimeError
        for matrix_name in data_registry.get_set(dataset_name).get_available_matrices():
            pipeline_config.data_config = [DataMatrixConfig(
                dataset_name,
                matrix_name,
                [],
                None,
                SplitConfig('unknown', {}, 0.2)
            )]

            data_transitions = run_data_pipelines(
                pipeline_config,
                data_event_dispatcher,
                is_always_running
            )
            assert len(data_transitions) == 0, \
                'did not expect data transitions for a configuration with an unknown splitter'

    # test skipping of configuration that raise a FileNotFoundError
    dataset_name = 'dataset'
    matrix_name = 'matrix with unknown file'
    dataset = create_dataset_with_dummy_matrix(
        io_tmp_dir,
        dataset_name,
        matrix_name,
        DatasetFileConfig('unknown.tsv',FileOptionsConfig(None, None, None, False, False))
    )
    # add dummy dataset to the data registry temporarily
    data_registry.registry[dataset_name] = dataset

    pipeline_config.data_config = [DataMatrixConfig(
        dataset_name,
        matrix_name,
        [],
        None,
        create_default_split_config()
    )]

    data_transitions = run_data_pipelines(
        pipeline_config,
        data_event_dispatcher,
        is_always_running
    )
    assert len(data_transitions) == 0, \
        'did not expect data transitions for a configuration with a dataset without a matrix file'
    # cleanup dummy dataset from data registry
    del data_registry.registry[dataset_name]


def test_run_data_pipelines(
        io_tmp_dir: str,
        data_registry: DataRegistry,
        data_event_dispatcher: EventDispatcher) -> None:
    """Test the data pipeline (run) integration."""
    pipeline_config = DataPipelineConfig(
        io_tmp_dir,
        data_registry,
        create_data_factory(data_registry),
        create_data_matrix_config_list(data_registry)
    )

    data_transitions = run_data_pipelines(
        pipeline_config,
        data_event_dispatcher,
        is_always_running
    )
    assert len(pipeline_config.data_config_list) == len(data_transitions), \
        'expected data transition for each data matrix configuration'

    data_directories = os.listdir(io_tmp_dir)
    assert len(data_directories) == len(data_transitions), \
        'expected data directory for each data transition'

    for data_dir in data_directories:
        data_dir = os.path.join(io_tmp_dir, data_dir)
        assert os.path.isfile(os.path.join(data_dir, 'train_set.tsv')), \
            'expected saved train set in data transition output directory'
        assert os.path.isfile(os.path.join(data_dir, 'test_set.tsv')), \
            'expected saved test set in data transition output directory'


def create_data_matrix_config_list(datasets_registry: DataRegistry) -> List[DataMatrixConfig]:
    """Create data matrix configuration list for each available dataset matrix."""
    config_list = []

    for dataset_name in datasets_registry.get_available_sets():
        if not 'Sample' in dataset_name:
            continue

        dataset = datasets_registry.get_set(dataset_name)

        for matrix_name in dataset.get_available_matrices():
            config_list.append(DataMatrixConfig(
                dataset.get_name(),
                matrix_name,
                [],
                ConvertConfig(CONVERTER_RANGE, {'upper_bound': RATING_TYPE_THRESHOLD}),
                create_default_split_config()
            ))

    return config_list
