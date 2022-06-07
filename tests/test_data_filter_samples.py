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
    dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    filter_dataset_factory = filter_factory.get_factory(dataset_name).get_factory(matrix_name)
    config_list = filter_dataset_factory.get_available()
    for input in config_list:
        col_name = input['name']
        if 'gender' not in col_name.split('_')[-1] and 'country' not in col_name.split('_')[-1]:
            continue
        conditions = input['params']['options'][0]['default']
        res = []
        for condition in conditions:
            filterobj = filter_dataset_factory.create(col_name, {'values': [condition]})
            df_x = filterobj.run(dataframe)
            res.append(len(df_x))
        assert len(dataframe) == sum(res)

