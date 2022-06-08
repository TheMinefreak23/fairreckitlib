"""This module tests the filter factory on dataset samples.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
import pandas
from pandas.testing import assert_frame_equal
import pytest
from src.fairreckitlib.data.filter.filter_factory import create_filter_factory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.data.set.dataset import add_dataset_columns


dataset_registry = DataRegistry('tests/datasets')
movie_matrices = [
    ('ML-100K-Sample', 'user-movie-rating'),
    ('ML-25M-Sample', 'user-movie-rating'),
]
artist_matrices = [
    ('LFM-360K-Sample', 'user-artist-count'),
    ('LFM-1B-Sample', 'user-artist-count'),
    ('LFM-2B-Sample', 'user-artist-count'),
]
track_matrices = [
    ('LFM-2B-Sample', 'user-track-count'),
]
dataset_matrices = movie_matrices + artist_matrices + track_matrices

filter_factory = create_filter_factory(dataset_registry)

@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
def test_categorical_filter(dataset_name, matrix_name):
    """Test categorical filter on unique and total number of values per condition and as a whole.

    Tests all sample datasets. Default options should contain all values,
    i.e. the aggregation of the conditions equals the original dataframe.
    """
    og_dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    filter_dataset_factory = filter_factory.get_factory(dataset_name).get_factory(matrix_name)
    data_config_list = filter_dataset_factory.get_available()
    for data_config in data_config_list:
        col_name = data_config['name']
        if col_name.split('_')[-1] not in ['gender', 'country']:
            continue
        conditions = data_config['params']['options'][0]['default']
        df_lengths = []
        unique_vals = []
        for condition in conditions:
            filterobj = filter_dataset_factory.create(col_name, {'values': [condition]})
            df_x = filterobj.run(og_dataframe)
            df_lengths.append(len(df_x))
            df_x_with_col = add_dataset_columns(dataset_registry.get_set(dataset_name),
                                                matrix_name,
                                                df_x,
                                                [col_name]
                                                )
            unique_vals.append(len(pandas.unique(df_x_with_col[col_name])))
        assert len(og_dataframe) == sum(df_lengths)
        og_df = add_dataset_columns(dataset_registry.get_set(dataset_name),
                                    matrix_name,
                                    og_dataframe,
                                    [col_name]
                                    )
        assert og_df[col_name].nunique(False) == sum(unique_vals)

@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
def test_numerical_filter(dataset_name, matrix_name):
    """Test numerical filter on range (min, max) before and after run.

    Tests all sample datasets. The default values should include all entries of
    the original dataframe with the same range.
    """
    og_dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    filter_dataset_factory = filter_factory.get_factory(dataset_name).get_factory(matrix_name)
    data_config_list = filter_dataset_factory.get_available()
    for data_config in data_config_list:
        col_name = data_config['name']
        if col_name.split('_')[-1] not in ['age', 'rating', 'timestamp']:
            continue
        default_vals = data_config['params']['values'][0]['default']
        filterobj = filter_dataset_factory.create(col_name, {'values': default_vals})
        df_x = filterobj.run(og_dataframe)
        df_x_with_col = add_dataset_columns(dataset_registry.get_set(dataset_name),
                                            matrix_name,
                                            df_x,
                                            [col_name]
                                            )
        og_df_with_col = add_dataset_columns(dataset_registry.get_set(dataset_name),
                                             matrix_name,
                                             og_dataframe,
                                             [col_name]
                                             )
        assert ((og_df_with_col[col_name].min(), og_df_with_col[col_name].max()) ==
        (df_x_with_col[col_name].min(), df_x_with_col[col_name].max()))
        assert_frame_equal(df_x, og_dataframe)

@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
def test_count_filter(dataset_name, matrix_name):
    """Test count filter on thresholds: 0, default, infinite.

    Tests all sample datasets. The length of the dataframe using default threshold should be
    between thresholds infinite and 0.
    """
    og_dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    filter_dataset_factory = filter_factory.get_factory(dataset_name).get_factory(matrix_name)
    data_config_list = filter_dataset_factory.get_available()

    def short_assertion(col_name, test_int, test_threshold):
        filterobj = filter_dataset_factory.create(col_name, {'threshold': test_threshold})
        df_x = filterobj.run(og_dataframe)
        assert test_int >= len(df_x)

    for data_config in data_config_list:
        if data_config['name'].split('_')[-1] not in ['count']:
            continue
        col_name = data_config['name']

        test_scenarios = [(len(og_dataframe), 0),
                          (len(og_dataframe), data_config['params']['values'][0]['default']),
                          (0, math.inf)
                          ]
        for i, threshold in test_scenarios:
            short_assertion(col_name, i, threshold)
