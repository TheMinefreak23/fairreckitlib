"""This modules contains the class to process the MovieLens-100K dataset.

Classes:

    DataProcessorML100K: data processor implementation for the ML-100K dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import numpy as np

from ...utility import load_table
from ..dataset import DATASET_RATINGS_EXPLICIT
from ..dataset_constants import DATASET_PREFIX, DATASET_FILE
from ..dataset_table import create_table_config, read_table, write_table
from .processor_base import DataProcessorBase

MOVIE_GENRES = [
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


class DataProcessorML100K(DataProcessorBase):
    """DataProcessor for the MovieLens-100K dataset."""

    def __init__(self, dataset_name):
        """Construct the ML-100K DataProcessor.

        Args:
            dataset_name: name of the dataset.
        """
        DataProcessorBase.__init__(self, dataset_name, 'u.data')

    def load_matrix(self, file_path: str) -> Tuple[bool, str]:
        """Load the matrix of the dataset.

        The user-item matrix is stored in a tsv file. Only the
        ratings are converted to floating-point before returning.

        Args:
            file_path: the path to the matrix file.

        Returns:
            True, DATASET_RATINGS_EXPLICIT
        """
        self.data_matrix = load_table(
            file_path,
            ['user', 'item', 'rating', 'timestamp']
        )

        # convert original int ratings
        self.data_matrix['rating'] = self.data_matrix['rating'].astype(float)

        return True, DATASET_RATINGS_EXPLICIT

    def load_tables_config(self, dataset_dir: str, user_item_tables: Dict[str, Dict[str, Any]]):
        """Return the user and movie configurations."""
        return user_item_tables

    def load_user_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the user table of the dataset.

        Changes the contents of the gender and occupation columns to be more user-friendly.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the user table.
        """
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

        # load original user table
        user_table = read_table(
            dataset_dir,
            user_table_config
        )

        # convert gender and occupation to more user-friendly names
        user_table['user_gender'].replace({'M': 'Male', 'F': 'Female'}, inplace=True)
        user_table['user_occupation'] = user_table['user_occupation'].str.capitalize()

        # update user table configuration
        user_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_users.tsv'
        user_table_config['sep'] = None

        # store the generated user table
        write_table(
            user_table,
            dataset_dir,
            user_table_config[DATASET_FILE]
        )

        return 'user', user_table_config

    def load_item_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the movie table of the dataset.

        Simplifies the binary genre columns by concatenating the names using pipes.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the movie table.
        """
        movie_columns = [
            'movie_title',
            'movie_release date',
            'movie_video release date',
            'movie_imdb url'
        ]

        # create original table definition
        movie_table_config = create_table_config(
            'u.item',
            ['movie_id'],
            movie_columns + MOVIE_GENRES,
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
            columns=MOVIE_GENRES
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

        # store the generated movie table
        write_table(
            movie_table,
            dataset_dir,
            movie_table_config[DATASET_FILE],
            encoding=movie_table_config['encoding']
        )

        return 'movie', movie_table_config
