"""This module tests the dataframe rating conversion functionality.

Functions:

    test_converter_factory: test if factories and converters are created correctly.
    test_apc_alc: test is listen count <= play count.
    test_to_explicit: test if ratings are converted correctly.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pytest

from src.fairreckitlib.core.config.config_factories import GroupFactory
from src.fairreckitlib.data.data_modifier import DataModifierFactory
from src.fairreckitlib.data.ratings.base_converter import RatingConverter
from src.fairreckitlib.data.ratings.convert_constants import CONVERTER_KL, CONVERTER_RANGE
from src.fairreckitlib.data.ratings.kl_converter import KLConverter
from src.fairreckitlib.data.ratings.range_converter import RangeConverter
from src.fairreckitlib.data.ratings.rating_converter_factory import create_rating_converter_factory
from src.fairreckitlib.data.ratings import count
from src.fairreckitlib.data.set.dataset_registry import DataRegistry

# dataset matrices to run rating converters with
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

converter_factory = create_rating_converter_factory(dataset_registry)
rating_modifiers = [1.0, 5.0, 10.0, 1000.0]


def test_converter_factory():
    """Test all dataset/matrix group factories and rating converters in the converter factory."""
    assert isinstance(converter_factory, GroupFactory), \
        'expected converter factory to be a group of dataset factories'
    for dataset_name in converter_factory.get_available_names():
        assert dataset_name in dataset_registry.get_available_sets(), \
            'expected dataset converter factory to be available for ' + dataset_name

        dataset_converter_factory = converter_factory.get_factory(dataset_name)
        assert isinstance(dataset_converter_factory, GroupFactory), \
            'expected dataset converter factory to be a group of matrix factories'

        for matrix_name in dataset_converter_factory.get_available_names():
            assert matrix_name in dataset_registry.get_set(dataset_name).get_available_matrices(),\
                'expected matrix converter factory to be available for ' + matrix_name

            matrix_converter_factory = dataset_converter_factory.get_factory(matrix_name)
            assert isinstance(matrix_converter_factory, DataModifierFactory), \
                'expected matrix converter factory to be a data modifier factory'

            for converter_name in matrix_converter_factory.get_available_names():
                converter = matrix_converter_factory.create(converter_name)
                assert isinstance(converter, RatingConverter), \
                    'expected converter to be a derivative of a rating converter'

            converter = matrix_converter_factory.create(CONVERTER_RANGE)
            assert bool(converter) and isinstance(converter, RangeConverter), \
                'expected the range converter to be present'
            converter = matrix_converter_factory.create(CONVERTER_KL)
            assert bool(converter) and isinstance(converter, KLConverter), \
                'expected the kl converter to be present'

@pytest.mark.parametrize('dataset_name, matrix_name', artist_matrices)

def test_apc_alc(dataset_name: str, matrix_name: str) -> None:
    """Test if alc is always <= apc."""
    print('Testing APC/ALC for', dataset_name, matrix_name)

    dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    play = count.calculate_apc(dataframe)
    listen = count.calculate_alc(dataframe)
    for key, value in listen.items():
        assert value <= play[key], 'Listener count cannot be greater than playcount: ' \
            + str(key) + ' ' + str(value) + ' ' + str(play[key]) + ' ' + \
            dataset_name + ' ' + matrix_name

@pytest.mark.parametrize('dataset_name, matrix_name', dataset_matrices)
@pytest.mark.parametrize('modifier', rating_modifiers)

def test_range_converter(dataset_name, matrix_name, modifier):
    """Test if the ratings are converted to a range of [0...modifier]."""
    print('Testing', CONVERTER_RANGE, 'converter for',
          dataset_name, matrix_name, '=> upper bound', modifier)

    dataset_converter_factory = converter_factory.get_factory(dataset_name)
    matrix_converter_factory = dataset_converter_factory.get_factory(matrix_name)
    converter = matrix_converter_factory.create(CONVERTER_RANGE, {'upper_bound': modifier})
    dataframe = dataset_registry.get_set(dataset_name).load_matrix(matrix_name)
    (converted_df, _) = converter.run(dataframe)
    for _, row in converted_df.iterrows():
        assert 0 < row['rating'] <= modifier, \
            f'Rating {0} should be 0<x<{1} : {2}'.format(row['rating'],
                                                        str(modifier),
                                                        dataset_name + ' ' + matrix_name)
