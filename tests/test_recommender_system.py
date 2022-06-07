"""This module tests the top level functionality of the recommender system.

Functions:

    test_recommender_system_constructor: test the constructor of the recommender system.
    test_recommender_system_availability: test the availability of the recommender system.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_json
from src.fairreckitlib.core.io.io_delete import delete_dir, delete_file
from src.fairreckitlib.core.io.io_utility import load_json
from src.fairreckitlib.core.threading.thread_processor import ThreadProcessor
from src.fairreckitlib.recommender_system import RecommenderSystem
from .conftest import DATASET_DIR


def test_recommender_system_constructor(
        io_tmp_dir: str, io_event_dispatcher: EventDispatcher) -> None:
    """Test the constructor of the recommender system."""
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

    # clean up
    delete_dir(unknown_dir, io_event_dispatcher)


def test_recommender_system_availability(
        io_tmp_dir: str, io_event_dispatcher: EventDispatcher) -> None:
    """Test the availability functions of the recommender system."""
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
        json_path = os.path.join(io_tmp_dir, 'available_' + available_name + '.json')
        create_json(json_path, availability, io_event_dispatcher)
        assert availability == load_json(json_path), 'expected availability to be JSON compatible'
        delete_file(json_path, io_event_dispatcher)
