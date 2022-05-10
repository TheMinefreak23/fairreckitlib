"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
import pytest
from src.fairreckitlib.data.convert import convert_factory, to_explicit_converter, count
from src.fairreckitlib.data.set import dataset_registry

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers
# [user, item, artistname, rating]
df_lfm360k_sample = pd.read_csv(
    'tests\\datasets\\sample\\lfm-360k-sample.tsv', delimiter='\t')

datasets = dataset_registry.DataRegistry('tests/datasets')
set_ml_100k = datasets.get_set('ML-100K')
df_ml_100k = set_ml_100k.load_matrix_df()

dfs = [('ml_100k', df_ml_100k)
    ,  ('df_lfm360k', df_lfm360k_sample)
        ]



to_explicit = convert_factory.create_converter('to_explicit')

def test_convert_classes():
    """Tests if the created variables are in fact of that class."""
    assert isinstance(to_explicit, to_explicit_converter.ToExplicitConverter)

@pytest.mark.parametrize('data', dfs)

def test_apc_alc(data):
    """Tests if alc is always <= apc."""
    (df_name, df) = data
    play = count.calculate_apc(df)
    listen = count.calculate_alc(df)
    for key, value in listen.items():
        assert value <= play[key], 'Listener count cannot be greater than playcount: ' \
            + str(key) + ' ' + str(value) + ' ' + str(play[key]) + ' ' + df_name

@pytest.mark.parametrize('data', dfs)

def test_to_explicit(data):
    """Tests if the ratings are converted to an explicit range of [0,1]"""
    (df_name, df) = data
    converted_df = to_explicit.run(df)
    for _, row in converted_df.iterrows():
        assert row['rating'] == 0 or row['rating'] == 1, 'Rating should be 0 or 1: ' + df_name
