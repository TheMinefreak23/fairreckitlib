"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
import pytest

from src.fairreckitlib.data.convert import factory, to_explicit, count

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers and indices
# [user, artistid, artistname, plays]
df_lfm360k_sample = pd.read_csv(
    'tests\\datasets\\sample\\lfm-360k-sample.tsv', delimiter='\t')

dfs = [('df_lfm360k', df_lfm360k_sample)]

to_explicit_converter = factory.create_converter('to_explicit')

def test_convert_classes():
    """Tests if the created variables are in fact of that class."""
    assert isinstance(to_explicit_converter, to_explicit.ToExplicitConverter)

@pytest.mark.parametrize('data', dfs)

def test_to_explicit(data):
    """Tests if the ratings are converted to an explicit range of [0,1]"""
    (df_name, df) = data
    converted_df = to_explicit_converter.run(df)
    for _, row in converted_df.iterrows():
        assert row['rating'] == 0 or row['rating'] == 1, 'Rating should be 0 or 1: ' + df_name

@pytest.mark.parametrize('data', dfs)

def test_apc_alc(data):
    """Tests if alc is always <= apc."""
    (df_name, df) = data
    play = count.calculate_apc(df)
    listen = count.calculate_alc(df)
    for key, value in play.items():
        assert value >= listen[key], 'Listener count cannot be greater than playcount: ' \
            + str(key) + ' ' + str(value) + ' ' + str(listen[key]) + ' ' + df_name
