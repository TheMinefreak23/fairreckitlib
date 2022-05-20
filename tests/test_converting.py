"""This module tests the dataframe rating conversion functionality.

Functions:

    test_converter_factory: test if factories are created correctly.
    test_convert_classes: test if classes are created correctly.
    test_apc_alc: test is listen count <= play count.
    test_to_explicit: test if ratings are converted correctly.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd
import pytest

from src.fairreckitlib.core.factories import Factory
from src.fairreckitlib.data.ratings.base_converter import RatingConverter
from src.fairreckitlib.data.ratings import count, rating_converter_factory, range_converter
from src.fairreckitlib.data.set import dataset_registry

# sample of the first 1000 entries of the lfm-360k dataset
# this already has headers
# [user, item, artistname, rating]
df_lfm360k_sample = pd.read_csv(
    './tests/datasets/sample/lfm-360k-sample.tsv', delimiter='\t')

datasets = dataset_registry.DataRegistry('tests/datasets')
set_ml_100k = datasets.get_set('ML-100K')
df_ml_100k = set_ml_100k.load_matrix_df()

dfs = [('ml_100k', df_ml_100k)
    ,  ('df_lfm360k_sample', df_lfm360k_sample)
        ]



converter_factory = rating_converter_factory.create_rating_converter_factory()
rating_modifiers = [1, 5, 10, 1000]


def test_converter_factory():
    """Test if all converters in the factory are derived from the correct base class."""
    assert isinstance(converter_factory, Factory)
    for _, converter_name in enumerate(converter_factory.get_available_names()):
        converter = converter_factory.create(converter_name)
        assert isinstance(converter, RatingConverter)

def test_convert_classes():
    """Tests if the created variables are in fact of that class."""
    converter = converter_factory.create(rating_converter_factory.CONVERTER_RANGE)
    assert isinstance(converter, range_converter.RangeConverter)

@pytest.mark.parametrize('data', dfs)

def test_apc_alc(data):
    """Tests if alc is always <= apc."""
    (df_name, dataframe) = data
    play = count.calculate_apc(dataframe)
    listen = count.calculate_alc(dataframe)
    for key, value in listen.items():
        assert value <= play[key], 'Listener count cannot be greater than playcount: ' \
            + str(key) + ' ' + str(value) + ' ' + str(play[key]) + ' ' + df_name

@pytest.mark.parametrize('data', dfs)
@pytest.mark.parametrize('modifier', rating_modifiers)

def test_to_explicit(data, modifier):
    """Tests if the ratings are converted to an explicit range of [0,1]."""
    converter_params = {'upper_bound': modifier}
    converter = converter_factory.create(rating_converter_factory.CONVERTER_RANGE,
                                         converter_params)
    (df_name, dataframe) = data
    (converted_df, _) = converter.run(dataframe)
    for _, row in converted_df.iterrows():
        assert 0 < row['rating'] <= modifier, \
            f'Rating {0} should be 0<x<{1} : {2}'.format(row['rating'],
                                                        str(modifier),
                                                        df_name)
