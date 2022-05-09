"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import numpy as np

from ...utility import load_table
from ..dataset_config import DATASET_FILE
from ..dataset_config import DATASET_PREFIX
from ..dataset_config import DATASET_RATINGS_EXPLICIT
from ..dataset_table import create_table_config
from ..dataset_table import read_table
from ..dataset_table import write_table
from .processor_base import DataProcessorBase


class DataProcessorML100K(DataProcessorBase):
    """DataProcessor for the ML-100K dataset.

    Args:
        dataset_name(str): name of the dataset.
    """
    def __init__(self, dataset_name):
        DataProcessorBase.__init__(self, dataset_name, 'u.data')

    def load_matrix(self, file_path):
        self.data_matrix = load_table(
            file_path,
            ['user', 'item', 'rating', 'timestamp']
        )

        # convert original int ratings
        self.data_matrix['rating'] = self.data_matrix['rating'].astype(float)

        return True, DATASET_RATINGS_EXPLICIT

    def load_tables_config(self, dataset_dir, user_item_tables):
        return user_item_tables

    def load_user_table_config(self, dataset_dir):
        user_table_config = create_table_config(
            'u.user',
            ['user_id'],
            ['user_age',
             'user_gender',
             'user_occupation',
             'user_zip code'
             ],
            sep='|'
        )

        user_table = read_table(
            dataset_dir,
            user_table_config
        )

        user_table['user_gender'].replace({'M': 'Male', 'F': 'Female'}, inplace=True)
        user_table['user_occupation'] = user_table['user_occupation'].str.capitalize()

        user_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_users.tsv'
        user_table_config['sep'] = None

        write_table(
            user_table,
            dataset_dir,
            user_table_config[DATASET_FILE]
        )

        return 'user', user_table_config

    def load_item_table_config(self, dataset_dir):
        movie_columns = [
            'movie_title',
            'movie_release date',
            'movie_video release date',
            'movie_imdb url'
        ]

        movie_genres = [
            'Unknown',
            'Action',
            'Adventure',
            'Animation',
            'Children',
            'Comedy',
            'Crime',
            'Documentary',
            'Drama',
            'Fantasy',
            'Film-Noir',
            'Horror',
            'Musical',
            'Mystery',
            'Romance',
            'Sci-Fi',
            'Thriller',
            'War',
            'Western'
        ]

        # create original table definition
        movie_table_config = create_table_config(
            'u.item',
            ['movie_id'],
            movie_columns + movie_genres,
            sep='|',
            encoding='ISO-8859-1'
        )

        # read the original table without binary genres
        movie_table = read_table(
            dataset_dir,
            movie_table_config,
            columns=movie_table_config['keys'] + movie_columns
        )

        # read the binary genres table
        genres_table = read_table(
            dataset_dir,
            movie_table_config,
            columns=movie_genres
        )

        # replace 0 with NaN and 1 with the corresponding genre
        for column_name in genres_table:
            genres_table[column_name].replace({1:column_name, 0: np.nan}, inplace=True)

        # collapse genres into one column and add it to the original table
        genre_column = 'movie_genres'
        movie_table[genre_column] = genres_table.apply(lambda x: x.str.cat(sep='|'), axis=1)

        # update movie table definition
        movie_table_config['sep'] = None
        movie_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_movies.tsv'
        movie_table_config['columns'] = movie_columns + [genre_column]

        # safe the new movie table
        write_table(
            movie_table,
            dataset_dir,
            movie_table_config[DATASET_FILE],
            encoding=movie_table_config['encoding']
        )

        return 'movie', movie_table_config
