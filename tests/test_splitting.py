"""This module tests the splitting functionality.

Functions:

    test_split_factory: test split factory.
    test_split_classes: test split classes.
    test_temp_split: test temporal splitter.
    test_random_split: test random splitter.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
import pytest

from src.fairreckitlib.core.factories import Factory
from src.fairreckitlib.data.split.split_constants import DEFAULT_SPLIT_TEST_RATIO
from src.fairreckitlib.data.split.split_constants import KEY_SPLIT_TEST_RATIO
from src.fairreckitlib.data.split.split_constants import SPLIT_RANDOM, SPLIT_TEMPORAL
from src.fairreckitlib.data.split.split_factory import create_split_factory
from src.fairreckitlib.data.split.base_splitter import DataSplitter
from src.fairreckitlib.data.split.random_splitter import RandomSplitter
from src.fairreckitlib.data.split.temporal_splitter import TemporalSplitter

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers
# [user, item, artistname, rating]
df_lfm360k_sample = pd.read_csv(
    './tests/datasets/sample/lfm-360k-sample.tsv', delimiter='\t')

# sample of the first 1000 entries of the ml-100k dataset
# this already has headers
# [user, item, rating, timestamp]
df_ml100k_sample = pd.read_csv(
    './tests/datasets/sample/ml-100k-sample.tsv', delimiter='\t')

# the list of dataframes to test splitting with
dfs =  [('df_lfm360k', df_lfm360k_sample),
        ('df_ml100k' , df_ml100k_sample)]

# creating the factories to run splitting with
split_factory = create_split_factory()
split_kwargs = {KEY_SPLIT_TEST_RATIO: DEFAULT_SPLIT_TEST_RATIO}

# the list of test ratios to test splitting with
# should be a 0.0 < float < 1.0
ratios = [0.2, 0.3, 0.8]

def test_split_factory():
    """Test if all splitters in the factory are derived from the correct base class."""
    assert isinstance(split_factory, Factory)
    for _, splitter_name in enumerate(split_factory.get_available_names()):
        splitter = split_factory.create(splitter_name, None, **split_kwargs)
        assert isinstance(splitter, DataSplitter)

@pytest.mark.parametrize('splitter_name, splitter_type', [
    (SPLIT_RANDOM, RandomSplitter), (SPLIT_TEMPORAL, TemporalSplitter)
])

def test_split_classes(splitter_name, splitter_type):
    """Test if the created splitters are an isntance of that class."""
    splitter = split_factory.create(splitter_name, None, **split_kwargs)
    assert isinstance(splitter, splitter_type)

@pytest.mark.parametrize('data', dfs)
@pytest.mark.parametrize('ratio', ratios)

def test_temp_split(data, ratio):
    """Test if the temporal split returns a tuple with the correct ratio.

    Ratio has a 75% margin. A larger margin because it is split on user timestamps,
    which differs slightly from the overall timestamps in the dataset.
    """
    temp_split = split_factory.create(SPLIT_TEMPORAL, None, **{KEY_SPLIT_TEST_RATIO: ratio})
    (df_name, dataframe) = data
    if 'timestamp' in dataframe:
        (train, test) = temp_split.run(dataframe)
        assert len(train.index) != 0, \
            'Train set is empty: ' + df_name + str(ratio)
        assert len(test.index) != 0, \
            'Test set is empty: ' + df_name + str(ratio)

        ratio = len(test.index) / (len(train.index) + len(test.index))
        assert (ratio * 0.25) < ratio < (ratio * 1.75), \
            'Test set should be around ' + str(ratio) + ': '# + df_name

        # for every row, assert that the timestamps are bigger in the test set per user
        for _, row in train.iterrows():
            for _, row_ in test.iterrows():
                if row['user'] == row_['user']:
                    assert row['timestamp'] <= row_['timestamp']

@pytest.mark.parametrize('data', dfs)
@pytest.mark.parametrize('ratio', ratios)

def test_random_split(data, ratio):
    """Test if the random split returns a tuple with the correct raio.

    Ratio has a 10% margin.
    """
    random_split = split_factory.create(SPLIT_RANDOM, None, **{KEY_SPLIT_TEST_RATIO: ratio})
    (df_name, dataframe) = data
    (train, test) = random_split.run(dataframe)
    assert len(train.index) != 0, 'Train set is empty: ' + df_name + str(ratio)
    assert len(test.index) != 0, 'Test set is empty: ' + df_name + str(ratio)

    ratio = len(test.index) / (len(train.index) + len(test.index))
    assert (ratio * 0.9) < ratio < (ratio * 1.1), \
        'Test set should be around ' + str(ratio) + ': ' + df_name
