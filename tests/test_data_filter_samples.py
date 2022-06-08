"""This module tests the filter factory on dataset samples.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from numpy import clongdouble
import pandas
import pytest
import random
from pandas.testing import assert_frame_equal
from src.fairreckitlib.data.filter.base_filter import DataFilter

from src.fairreckitlib.data.filter.filter_factory import create_filter_factory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry

from src.fairreckitlib.data.set.dataset import add_dataset_columns	


random.seed(0)



dataset_registry = DataRegistry('datasets')
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
# {'user-artist-count': [
#     {'name': 'artist_gender', 'params': {'options': [{'name': 'values', 'options': [None, 'Male', 'Female'], 'default': [None, 'Male', 'Female']}], 'values': []}}, 
#     {'name': 'user_age', 'params': {'options': [], 'values': [{'name': 'range', 'min': -1, 'max': 57, 'default': {'min': -1, 'max': 57}}]}}, 
#     {'name': 'user_country', 'params': {'options': [{'name': 'values', 'options': ['DE', 'CA', 'MX', 'US', 'GB', 'FI', 'PL', 'ES', 'UA', 'SE', 'JP', 'AU', 'PT', 'RU', 'BE', 'GT', 'NL', 'BG', 'IT', 'AT', 'KR', 'BR', 'ZA', 'CZ', 'GR', 'IR', 'CL', 'TH', 'NO', 'SK', 'TW', 'TR', 'CY'], 'default': ['DE', 'CA', 'MX', 'US', 'GB', 'FI', 'PL', 'ES', 'UA', 'SE', 'JP', 'AU', 'PT', 'RU', 'BE', 'GT', 'NL', 'BG', 'IT', 'AT', 'KR', 'BR', 'ZA', 'CZ', 'GR', 'IR', 'CL', 'TH', 'NO', 'SK', 'TW', 'TR', 'CY']}], 'values': []}}, 
#     {'name': 'user_country_count', 'params': {'options': [], 'values': [{'name': 'threshold', 'min': 1, 'max': 10000000000, 'default': 100}]}}, 
#     {'name': 'user_gender', 'params': {'options': [{'name': 'values', 'options': ['Female', None, 'Male'], 'default': ['Female', None, 'Male']}], 'values': []}}
# ]}

[{'name': 'user_age', 'params': {'options': [], 'values': [{'name': 'range', 'min': -1, 'max': 58, 'default': {'min': -1, 'max': 58}}]}}, 
{'name': 'user_country', 'params': {'options': [{'name': 'values', 'options': ['US', 'NZ', 'NL', 'IL', 'DE', 'BR', 'FI', 'UK', 'CZ', 'BY', 'IT', 'PL', 'CA', None, 'RU', 'ES', 'FR', 'SE', 'RO', 'LT', 'AU', 'BE', 'MX', 'HR', 'CL', 'JP', 'SK', 'NO', 'UA', 'CO', 'UY', 'HU', 'BO', 'PT', 'EE'], 'default': ['US', 'NZ', 'NL', 'IL', 'DE', 'BR', 'FI', 'UK', 'CZ', 'BY', 'IT', 'PL', 'CA', None, 'RU', 'ES', 'FR', 'SE', 'RO', 'LT', 'AU', 'BE', 'MX', 'HR', 'CL', 'JP', 'SK', 'NO', 'UA', 'CO', 'UY', 'HU', 'BO', 'PT', 'EE']}], 'values': []}}, 
{'name': 'user_country_count', 'params': {'options': [], 'values': [{'name': 'threshold', 'min': 1, 'max': 10000000000, 'default': 100}]}}, 
{'name': 'user_gender', 'params': {'options': [{'name': 'values', 'options': ['Female', 'Male', 'Neutral', None], 'default': ['Female', 'Male', 'Neutral', None]}], 'values': []}}]

# 

@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
def test_categorical_filter(dataset_name, matrix_name):
    """Test categorical filter on unique and total number of values per condition and as a whole.
    
    Tests all sample datasets. Default options should contain all values, 
    i.e. the aggregation of the conditions equals the original dataframe.
    """

    og_dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    filter_dataset_factory = filter_factory.get_factory(dataset_name).get_factory(matrix_name)
    data_config_list = filter_dataset_factory.get_available()
    for input in data_config_list:
        col_name = input['name']
        if col_name.split('_')[-1] not in ['gender', 'country']:
            continue
        conditions = input['params']['options'][0]['default']
        df_lengths = []
        unique_vals = []
        for condition in conditions:
            filterobj = filter_dataset_factory.create(col_name, {'values': [condition]})
            df_x = filterobj.run(og_dataframe)
            df_lengths.append(len(df_x))
            df_x_with_col = add_dataset_columns(dataset_registry.get_set(dataset_name), matrix_name, df_x, [col_name])
            unique_vals.append(len(pandas.unique(df_x_with_col[col_name])))
        assert len(og_dataframe) == sum(df_lengths)
        og_df = add_dataset_columns(dataset_registry.get_set(dataset_name), matrix_name, og_dataframe, [col_name])
        assert og_df[col_name].nunique(False) == sum(unique_vals)

@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
def test_numerical_filter(dataset_name, matrix_name):
    """Test categorical filter on unique and total number of values per condition and as a whole.
    
    Tests all sample datasets. The default values should include all entries of
    the original dataframe.
    """

    og_dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    filter_dataset_factory = filter_factory.get_factory(dataset_name).get_factory(matrix_name)
    data_config_list = filter_dataset_factory.get_available()
    for input in data_config_list:
        col_name = input['name']
        if col_name.split('_')[-1] not in ['age', 'rating', 'timestamp']:
            continue
        default_vals = input['params']['values'][0]['default']
        filterobj = filter_dataset_factory.create(col_name, {'values': default_vals})
        df_x = filterobj.run(og_dataframe)
        df_x_with_col = add_dataset_columns(dataset_registry.get_set(dataset_name), matrix_name, df_x, [col_name])
        og_df_with_col = add_dataset_columns(dataset_registry.get_set(dataset_name), matrix_name, og_dataframe, [col_name])
        assert (og_df_with_col[col_name].min(), og_df_with_col[col_name].max()) == (df_x_with_col[col_name].min(), df_x_with_col[col_name].max())

