"""This module tests the top level functionality of the recommender system.

Functions:

    test_recommender_system_constructor: test the constructor of the recommender system.
    test_recommender_system_run_experiment_failures: test run experiment failures.
    test_recommender_system_run_multiple_experiments: test running multiple experiments.
    test_recommender_system_validate_experiment: test validating a previously computed experiment.
    test_recommender_system_abort_experiment: test aborting an experiment within reasonable time.
    test_recommender_system_availability: test the availability of the recommender system.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time

import pytest

from src.fairreckitlib.core.core_constants import VALID_TYPES
from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_dir, create_json, create_yml
from src.fairreckitlib.core.io.io_delete import delete_dir, delete_file
from src.fairreckitlib.core.io.io_utility import load_json
from src.fairreckitlib.core.threading.thread_processor import ThreadProcessor
from src.fairreckitlib.experiment.experiment_event import \
    ON_END_EXPERIMENT_THREAD, ExperimentThreadEventArgs
from src.fairreckitlib.recommender_system import RecommenderSystem
from .conftest import DATASET_DIR, NUM_THREADS
from .test_experiment_pipeline import \
    create_experiment_config_coverage, create_experiment_config_duplicates
from .test_experiment_thread import assert_experiment_thread_success


def test_recommender_system_constructor(io_tmp_dir: str) -> None:
    """Test the constructor of the recommender system."""
    print('\nRecommender system construction tests\n')
    unknown_dir = os.path.join(io_tmp_dir, 'unknown')
    # test failure on unknown dataset directory
    pytest.raises(IOError, RecommenderSystem, unknown_dir, io_tmp_dir)

    # test success on creating result dir
    RecommenderSystem(DATASET_DIR, unknown_dir)
    assert os.path.isdir(unknown_dir), 'expected result directory to be created'

    # test success on existing result dir
    recommender_system = RecommenderSystem(DATASET_DIR, unknown_dir)
    assert os.path.isdir(unknown_dir), 'expected result directory to be present'

    assert isinstance(recommender_system.experiment_factory, GroupFactory), \
        'expected experiment factory to be created on construction'

    assert isinstance(recommender_system.thread_processor, ThreadProcessor), \
        'expected thread processor to be created on construction'

    assert len(recommender_system.get_active_computations()) == 0, \
        'did not expect any active computations on construction'


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_recommender_system_run_experiment_failures(
        experiment_type: str,
        io_tmp_dir: str,
        io_event_dispatcher: EventDispatcher) -> None:
    """Test running experiments to fail for various errors."""
    print('\nRecommender system failed', experiment_type, 'experiment tests\n')
    recommender_system = RecommenderSystem(DATASET_DIR, io_tmp_dir)

    # test failure for incorrect experiment configurations
    for experiment_config in [None, True, False, 0, 0.0, 'a', [], {}, {'set'}]:
        pytest.raises(TypeError, recommender_system.run_experiment, experiment_config)

    experiment_config, _ = create_experiment_config_coverage(
        experiment_type,
        recommender_system.data_registry,
        recommender_system.experiment_factory
    )

    # test failure for an already existing result directory
    result_dir = os.path.join(io_tmp_dir, experiment_config.name)
    create_dir(result_dir, io_event_dispatcher)
    pytest.raises(IOError, recommender_system.run_experiment, experiment_config)
    delete_dir(result_dir, io_event_dispatcher)

    # test failure when parsing does not succeed
    experiment_config.datasets = [] # mandatory and will fail parsing
    assert not recommender_system.run_experiment(experiment_config, validate_config=True), \
        'expected experiment configuration validation to fail'
    assert not recommender_system.abort_computation(experiment_config.name), \
        'did not expect to abort a computation that failed configuration validation'
    assert len(recommender_system.get_active_computations()) == 0, \
        'did not expect any active computations after the validation failed'

    # test failure when parsing does not succeed while loading a yml file
    yml_path = os.path.join(io_tmp_dir, experiment_config.name + '_config')
    create_yml(yml_path + '.yml', experiment_config.to_yml_format(), io_event_dispatcher)
    assert not recommender_system.run_experiment_from_yml(yml_path), \
        'expected experiment configuration parsing to fail from a yml file'
    assert not recommender_system.abort_computation(experiment_config.name), \
        'did not expect to abort a computation that failed configuration parsing'
    assert len(recommender_system.get_active_computations()) == 0, \
        'did not expect any active computations after the parsing failed'

    # test failure when the yml file does not exist
    delete_file(yml_path + '.yml', io_event_dispatcher)
    pytest.raises(FileNotFoundError, recommender_system.run_experiment_from_yml, yml_path)


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_recommender_system_run_multiple_experiments(
        experiment_type: str,
        io_tmp_dir: str,
        io_event_dispatcher: EventDispatcher) -> None:
    """Test running multiple experiments at the same time."""
    print('\nRecommender system multiple', experiment_type, 'experiment tests\n')
    recommender_system = RecommenderSystem(DATASET_DIR, io_tmp_dir)

    configurations = [
        create_experiment_config_coverage(
            experiment_type,
            recommender_system.data_registry,
            recommender_system.experiment_factory
        )[0],
        create_experiment_config_duplicates(
            experiment_type,
            recommender_system.data_registry,
            recommender_system.experiment_factory
        )
    ]

    for i, experiment_config in enumerate(configurations):
        # make the computation names distinctive
        experiment_config.name += str(i)
        yml_path = os.path.join(io_tmp_dir, experiment_config.name + '_config' + str(i))
        create_yml(yml_path + '.yml', experiment_config.to_yml_format(), io_event_dispatcher)
        # run through yml covers integration as well
        assert recommender_system.run_experiment_from_yml(
            yml_path,
            num_threads=NUM_THREADS,
            events={ON_END_EXPERIMENT_THREAD: assert_experiment_thread_success},
            verbose=False)
        assert recommender_system.thread_processor.get_num_active() == i + 1, \
            'expected experiment to have started'

    # wait for the experiments to finish
    while recommender_system.thread_processor.get_num_active() > 0:
        time.sleep(1)


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_recommender_system_validate_experiment(
        experiment_type: str,
        io_tmp_dir: str,
        io_event_dispatcher: EventDispatcher) -> None:
    """Test validating a previously computed experiment."""
    print('\nRecommender system validate', experiment_type, 'experiment tests\n')
    recommender_system = RecommenderSystem(DATASET_DIR, io_tmp_dir)

    experiment_config = create_experiment_config_duplicates(
        experiment_type,
        recommender_system.data_registry,
        recommender_system.experiment_factory,
        num_data_duplicates=1,
        num_model_duplicates=1,
        num_metric_duplicates=1
    )

    # test failure when the previously computed experiment does not exist
    pytest.raises(IOError, recommender_system.validate_experiment, experiment_config.name, 1)

    yml_path = os.path.join(io_tmp_dir, experiment_config.name + '_config')
    create_yml(yml_path + '.yml', experiment_config.to_yml_format(), io_event_dispatcher)
    # run through yml covers integration as well
    assert recommender_system.run_experiment_from_yml(
        yml_path,
        events={ON_END_EXPERIMENT_THREAD: assert_experiment_thread_success},
        num_threads=NUM_THREADS,
        verbose=False), \
        'expected running the experiment to succeed for a valid configuration'
    assert len(recommender_system.get_active_computations()) == 1 and \
        experiment_config.name in recommender_system.get_active_computations(), \
        'expected the experiment computation to have started'

    # test failure to validate experiment that is currently being computed
    pytest.raises(KeyError, recommender_system.validate_experiment, experiment_config.name, 1)

    # wait for the experiments to finish
    while recommender_system.thread_processor.get_num_active() > 0:
        time.sleep(1)

    # test failure for previously computed experiment that is missing a configuration file
    config_path = os.path.join(io_tmp_dir, experiment_config.name, 'config.yml')
    delete_file(config_path, io_event_dispatcher)
    pytest.raises(FileNotFoundError,
                  recommender_system.validate_experiment, experiment_config.name, 1)
    create_yml(config_path, experiment_config.to_yml_format(), io_event_dispatcher)

    # test successfully starting experiment validation
    assert recommender_system.validate_experiment(
        experiment_config.name,
        2,
        events={ON_END_EXPERIMENT_THREAD: assert_experiment_thread_success},
        num_threads=NUM_THREADS,
        verbose=False), \
        'expected validating the experiment to succeed for a previously computed experiment'

    # wait for the experiment validation to finish
    while recommender_system.thread_processor.get_num_active() > 0:
        time.sleep(1)


@pytest.mark.parametrize('experiment_type', VALID_TYPES)
def test_recommender_system_abort_experiment(
        experiment_type: str,
        io_tmp_dir: str,
        io_event_dispatcher: EventDispatcher) -> None:
    """Test aborting an experiment to finish within reasonable time."""
    print('\nRecommender system aborting', experiment_type, 'experiment tests\n')
    recommender_system = RecommenderSystem(DATASET_DIR, io_tmp_dir)

    experiment_config = create_experiment_config_duplicates(
        experiment_type,
        recommender_system.data_registry,
        recommender_system.experiment_factory,
        # high number of duplicates to simulate a long experiment with samples
        num_data_duplicates=5,
        num_model_duplicates=10,
        num_metric_duplicates=20
    )

    def assert_is_experiment_aborted(_, event_args: ExperimentThreadEventArgs, **__) -> None:
        """Assert whether the experiment was not running when it finished."""
        assert not event_args.is_running, \
            'did not expect the experiment to be running when the thread finished'

    yml_path = os.path.join(io_tmp_dir, experiment_config.name + '_config')
    create_yml(yml_path + '.yml', experiment_config.to_yml_format(), io_event_dispatcher)
    # run through yml covers integration as well
    assert recommender_system.run_experiment_from_yml(
        yml_path,
        events={ON_END_EXPERIMENT_THREAD: assert_is_experiment_aborted},
        num_threads=NUM_THREADS,
        verbose=False), \
        'expected running the experiment to succeed for a valid configuration'
    assert len(recommender_system.get_active_computations()) == 1 and \
        experiment_config.name in recommender_system.get_active_computations(), \
        'expected the experiment computation to have started'

    start_time = time.time()
    begin = start_time
    abort_time = 5 # the experiment configuration that is used, takes a lot longer to finish
    elapsed_time = abort_time
    # let it run for a while
    while elapsed_time >= 0:
        end = time.time()
        elapsed_time -= (end - begin)
        time.sleep(0.5)

    # request experiment to abort
    assert recommender_system.abort_computation(experiment_config.name), \
        'expected the experiment to be aborted while it is running'

    # give the thread some time to finish
    while recommender_system.thread_processor.get_num_active() > 0:
        time.sleep(0.1)

    # assert that the whole experiment did not complete
    assert time.time() - start_time < abort_time + 1, \
        'expected the experiment to finish earlier when aborted'


def test_recommender_system_availability(
        io_tmp_dir: str, io_event_dispatcher: EventDispatcher) -> None:
    """Test the availability functions of the recommender system."""
    print('\nRecommender system availability tests\n')
    recommender_system = RecommenderSystem(DATASET_DIR, io_tmp_dir)
    available_funcs = [
        ('algorithms', recommender_system.get_available_algorithms),
        ('datasets', recommender_system.get_available_datasets),
        ('data_filters', recommender_system.get_available_data_filters),
        ('metrics', recommender_system.get_available_metrics),
        ('rating_converters', recommender_system.get_available_rating_converters),
        ('splitters', recommender_system.get_available_splitters),
    ]

    for available_name, func_available in available_funcs:
        availability = func_available()
        # integration check to verify everything is still available
        assert len(availability) > 0, 'expected availability'

        # assert availability with JSON (which implies compatibility with YML as well)
        json_path = os.path.join(io_tmp_dir, 'available_' + available_name + '.json')
        create_json(json_path, availability, io_event_dispatcher)
        assert availability == load_json(json_path), 'expected availability to be JSON compatible'
        delete_file(json_path, io_event_dispatcher)
