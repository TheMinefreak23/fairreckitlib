"""This module tests the data registry.

Functions:

    test_data_registry_construction: test the construction of a data registry.
    test_data_registry_get_available_processors: test the available processors of a data registry.
    test_data_registry_get_available_sets: test the available sets of a data registry.
    test_data_registry_get_info: test dataset information of a data registry.
    test_data_registry_get_set: test the dataset retrieval from a data registry.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd
import pytest

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_dir
from src.fairreckitlib.data.set.dataset import Dataset
from src.fairreckitlib.data.set.dataset_config_parser import DatasetConfigParser
from src.fairreckitlib.data.set.dataset_constants import DATASET_CONFIG_FILE
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from .conftest import DATASET_DIR


def test_data_registry_construction(io_tmp_dir: str, io_event_dispatcher: EventDispatcher) -> None:
    """Test the construction of a data registry."""
    pytest.raises(IOError, DataRegistry, os.path.join(io_tmp_dir, 'unknown'))

    # test skipping of files in the data directory
    pd.DataFrame().to_csv(os.path.join(io_tmp_dir, 'file.tsv'))
    data_registry = DataRegistry(io_tmp_dir)
    assert len(data_registry.get_available_sets()) == 0, \
        'expected data registry to skip a file in the data directory on construction'

    available_processors = data_registry.get_available_processors()

    # test skipping of directories where the dataset processor is unknown
    create_dir(os.path.join(io_tmp_dir, 'unknown_dataset'), io_event_dispatcher)
    data_registry = DataRegistry(io_tmp_dir)
    assert len(data_registry.get_available_sets()) == 0, \
        'expected data registry to skip a directory in the data directory for an unknown processor'

    # test skipping of directories where the dataset processor failed
    for processor_name in available_processors:
        create_dir(os.path.join(io_tmp_dir, processor_name), io_event_dispatcher)
        data_registry = DataRegistry(io_tmp_dir)
        assert len(data_registry.get_available_sets()) == 0, \
            'expected dataset processing to fail for a valid directory without dataset files'

    # test success for a directory with preprocessed datasets
    data_config_parser = DatasetConfigParser(True)
    num_valid_dirs = 0
    for data_dir in os.listdir(DATASET_DIR):
        data_dir = os.path.join(DATASET_DIR, data_dir)
        if os.path.isfile(os.path.join(data_dir, DATASET_CONFIG_FILE)) and \
                data_config_parser.parse_dataset_config_from_yml(data_dir, DATASET_CONFIG_FILE,[]):
            num_valid_dirs += 1

    assert len(DataRegistry(DATASET_DIR).get_available_sets()) == num_valid_dirs, \
        'expected all valid datasets to be available in the data registry'


def test_data_registry_get_available_processors(data_registry: DataRegistry) -> None:
    """Test the retrieval of available dataset processors from a data registry."""
    available_processors = data_registry.get_available_processors()
    assert len(available_processors) == len(data_registry.processors), \
        'expected all registered dataset processors to be available'

    for dataset_processor_name in available_processors:
        assert dataset_processor_name in data_registry.processors, \
            'expected dataset processor name to be present in the available processors'


def test_data_registry_get_available_sets(data_registry: DataRegistry) -> None:
    """Test the retrieval of available dataset names from a data registry."""
    available_sets = data_registry.get_available_sets()
    assert len(available_sets) == len(data_registry.registry), \
        'expected all registered datasets to be available'

    for dataset_name in available_sets:
        assert dataset_name in data_registry.registry, \
            'expected dataset to be available in the registry'


def test_data_registry_get_info(data_registry: DataRegistry) -> None:
    """Test the retrieval of dataset information from a data registry."""
    info = data_registry.get_info()
    assert len(info) == len(data_registry.get_available_sets()), \
        'expected info for all available datasets'

    for dataset_name in data_registry.get_available_sets():
        assert dataset_name in info, \
            'expected dataset info to be present'


def test_data_registry_get_set(data_registry: DataRegistry) -> None:
    """Test the dataset retrieval from a data registry."""
    assert not bool(data_registry.get_set('unknown')), \
        'did not expect unknown dataset to be retrieved'

    for dataset_name in data_registry.get_available_sets():
        dataset = data_registry.get_set(dataset_name)

        assert bool(dataset), 'expected known dataset to be retrieved'
        assert isinstance(dataset, Dataset), 'expected dataset to be retrieved'
