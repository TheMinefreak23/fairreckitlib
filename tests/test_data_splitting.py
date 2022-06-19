"""This module tests the dataframe splitting functionality.

Classes:

    DummySplitter: dummy splitter to test not implemented errors.

Functions:

    test_splitter_interface_error: test interface error for not implemented functions.
    test_splitter_min_max_test_ratio: test min and max test ratio to be sufficient.
    test_split_factory: test split factory.
    test_split_classes: test split classes.
    test_temp_split: test temporal splitter.
    test_random_split: test random splitter.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Tuple

import pandas as pd
import pytest

from src.fairreckitlib.core.config.config_factories import Factory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.data.split.split_constants import DEFAULT_SPLIT_TEST_RATIO
from src.fairreckitlib.data.split.split_constants import KEY_SPLIT_TEST_RATIO
from src.fairreckitlib.data.split.split_constants import MIN_TEST_RATIO, MAX_TEST_RATIO
from src.fairreckitlib.data.split.split_constants import SPLIT_RANDOM, SPLIT_TEMPORAL
from src.fairreckitlib.data.split.split_factory import create_split_factory
from src.fairreckitlib.data.split.base_splitter import DataSplitter
from src.fairreckitlib.data.split.random_splitter import RandomSplitter
from src.fairreckitlib.data.split.temporal_splitter import TemporalSplitter

# dataset matrices to run splitting with
timestamp_matrices = [
    ('ML-100K-Sample', 'user-movie-rating'),
    ('ML-25M-Sample', 'user-movie-rating'),
]
dataset_matrices = timestamp_matrices + [
    ('LFM-360K-Sample', 'user-artist-count'),
    ('LFM-1B-Sample', 'user-artist-count'),
    ('LFM-2B-Sample', 'user-artist-count'),
    ('LFM-2B-Sample', 'user-track-count'),
]

# creating the factory to run splitting with
split_factory = create_split_factory()
split_kwargs = {KEY_SPLIT_TEST_RATIO: DEFAULT_SPLIT_TEST_RATIO}

# the list of test ratios to test splitting with
# should be a 0.0 < float < 1.0
ratios = [MIN_TEST_RATIO, DEFAULT_SPLIT_TEST_RATIO, 1.0 - DEFAULT_SPLIT_TEST_RATIO, MAX_TEST_RATIO]


class DummySplitter(DataSplitter):
    """Dummy splitter to test not implemented errors."""

    def __init__(self):
        """Construct dummy splitter."""
        DataSplitter.__init__(self, 'splitter', {}, 0.2)

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Raise NotImplementedError."""
        return DataSplitter.run(self, dataframe)


def test_splitter_interface_error() -> None:
    """Test data splitter interface errors for not implemented functions."""
    splitter = DummySplitter()

    pytest.raises(NotImplementedError, splitter.run, None)


def test_splitter_min_max_test_ratio() -> None:
    """Test the minimum and maximum test ratio to be sensible values."""
    assert MIN_TEST_RATIO > 0.0, 'expected minimum test ratio to be greater than zero'
    assert MAX_TEST_RATIO < 1.0, 'expected maximum test ratio to be less than one'


def test_split_factory():
    """Test if all splitters in the factory are derived from the correct base class."""
    assert isinstance(split_factory, Factory)
    for splitter_name in split_factory.get_available_names():
        splitter = split_factory.create(splitter_name, None, **split_kwargs)
        assert isinstance(splitter, DataSplitter)


@pytest.mark.parametrize('splitter_name, splitter_type', [
    (SPLIT_RANDOM, RandomSplitter), (SPLIT_TEMPORAL, TemporalSplitter)
])
def test_split_classes(splitter_name, splitter_type):
    """Test if the created splitters are an instance of that class."""
    splitter = split_factory.create(splitter_name, None, **split_kwargs)
    assert isinstance(splitter, splitter_type)

    # test failure on edge cases of the test ratio's accepted values
    with pytest.raises(RuntimeError):
        split_factory.create(splitter_name, None,
                             **{KEY_SPLIT_TEST_RATIO: MIN_TEST_RATIO - 0.00001})
        split_factory.create(splitter_name, None,
                             **{KEY_SPLIT_TEST_RATIO: MAX_TEST_RATIO + 0.00001})


@pytest.mark.parametrize('dataset_name, matrix_name', timestamp_matrices)
@pytest.mark.parametrize('ratio', ratios)
def test_temp_split(
        data_registry: DataRegistry, dataset_name: str, matrix_name: str, ratio: float) -> None:
    """Test if the temporal split returns a tuple with the correct ratio.

    Ratio has a 75% margin. A larger margin because it is split on user timestamps,
    which differs slightly from the overall timestamps in the dataset.
    """
    print('Testing', SPLIT_TEMPORAL, 'split for',
          dataset_name, matrix_name, '=> ratio', ratio)

    temp_split = split_factory.create(SPLIT_TEMPORAL, None, **{KEY_SPLIT_TEST_RATIO: ratio})
    dataframe = data_registry.get_set(dataset_name).load_matrix(matrix_name)
    assert 'timestamp' in dataframe, 'expected timestamp to be present in the dataset matrix'

    (train, test) = temp_split.run(dataframe)
    assert len(train.index) != 0, \
        'Train set is empty: ' + dataset_name + ' ' + matrix_name + str(ratio)
    assert len(test.index) != 0, \
        'Test set is empty: ' + dataset_name + ' ' + matrix_name + str(ratio)

    ratio = len(test.index) / (len(train.index) + len(test.index))
    assert (ratio * 0.25) < ratio < (ratio * 1.75), \
        'Test set should be around ' + str(ratio) + ': '# + df_name

    # for every row, assert that the timestamps are bigger in the test set per user
    for _, row in train.iterrows():
        for _, row_ in test.iterrows():
            if row['user'] == row_['user']:
                assert row['timestamp'] <= row_['timestamp']


@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
@pytest.mark.parametrize('ratio', ratios)
def test_random_split(
        data_registry: DataRegistry, dataset_name: str, matrix_name: str, ratio: float) -> None:
    """Test if the random split returns a tuple with the correct raio.

    Ratio has a 10% margin.
    """
    print('Testing', SPLIT_RANDOM, 'split for',
          dataset_name, matrix_name, '=> ratio', ratio)

    random_split = split_factory.create(SPLIT_RANDOM, None, **{KEY_SPLIT_TEST_RATIO: ratio})
    dataframe = data_registry.get_set(dataset_name).load_matrix(matrix_name)
    (train, test) = random_split.run(dataframe)
    assert len(train.index) != 0, \
        'Train set is empty: ' + dataset_name + ' ' + matrix_name + str(ratio)
    assert len(test.index) != 0, \
        'Test set is empty: ' + dataset_name + ' ' + matrix_name + str(ratio)

    ratio = len(test.index) / (len(train.index) + len(test.index))
    assert (ratio * 0.9) < ratio < (ratio * 1.1), \
        'Test set should be around ' + str(ratio) + ': ' + dataset_name + ' ' + matrix_name
