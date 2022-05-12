"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
import pytest
from src.fairreckitlib.data.ratings import count, rating_converter_factory, range_converter
from src.fairreckitlib.data.set import dataset_registry
from src.fairreckitlib.data.pipeline.data_pipeline import DataPipeline

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers
# [user, item, artistname, rating]
df_lfm360k_sample = pd.read_csv(
    'tests\\datasets\\sample\\lfm-360k-sample.tsv', delimiter='\t')

datasets = dataset_registry.DataRegistry('tests/datasets')
set_ml_100k = datasets.get_set('ML-100K')
df_ml_100k = set_ml_100k.load_matrix_df()

dfs = [('ml_100k', df_ml_100k)
    ,  ('df_lfm360k_sample', df_lfm360k_sample)
        ]



converter_factory = rating_converter_factory.create_rating_converter_factory()
data_pipeline = DataPipeline(None, None)
rating_modifier = 5

def test_convert_classes():
    """Tests if the created variables are in fact of that class."""
    converter = converter_factory.create(rating_converter_factory.CONVERTER_RANGE)
    assert isinstance(converter, range_converter.RangeConverter)

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
    converter_params = {'upper_bound': rating_modifier}
    converter = converter_factory.create(rating_converter_factory.CONVERTER_RANGE, converter_params)
    (df_name, df) = data
    (converted_df, rating_type) = converter.run(df)
    for _, row in converted_df.iterrows():
        assert 0 < row['rating'] <= rating_modifier, \
            'Rating {0} should be 0<x<{1} : {2}'.format(row['rating'], str(rating_modifier), df_name)
