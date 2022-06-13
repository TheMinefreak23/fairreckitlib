"""This modules contains the class to process the MovieLens-100K dataset.

Classes:

    DatasetProcessorML100K: data processor implementation for the ML-100K dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..dataset_config import DatasetTableConfig
from ..dataset_config import create_dataset_table_config
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_ml import DatasetProcessorML

MOVIE_GENRES = [
    'Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]


class DatasetProcessorML100K(DatasetProcessorML):
    """DatasetProcessor for the MovieLens-100K dataset.

    The dataset can be downloaded from the link below.
    https://files.grouplens.org/datasets/movielens/ml-100k.zip

    The processor handles the following files:

    u.data (required)
    u.user (optional)
    u.item (optional)
    """

    def create_user_movie_matrix_config(self) -> DatasetTableConfig:
        """Create the user-movie matrix configuration.

        Returns:
            the table configuration of the ML-100K matrix.
        """
        return create_dataset_table_config(
            'u.data',
            ['user_id', 'movie_id'],
            ['matrix_rating', 'matrix_timestamp'],
            foreign_keys=['user_id', 'movie_id']
        )

    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Returns:
            a list containing the user and movie table processors.
        """
        return [
            ('movie', self.process_movie_table),
            ('user', self.process_user_table)
        ]

    def process_movie_table(self) -> Optional[DatasetTableConfig]:
        """Process the movie table.

        Removes an empty release date column that is included in the movie title.
        Simplifies the binary genre columns by concatenating the names using pipes.

        Returns:
            the movie table configuration or None on failure.
        """
        movie_columns = [
            'movie_title',
            'movie_release date',
            'empty', # this column does not contain any data
            'movie_imdb url'
        ]

        # create original table definition
        movie_table_config = create_dataset_table_config(
            'u.item',
            ['movie_id'],
            movie_columns + MOVIE_GENRES,
            sep='|',
            encoding='ISO-8859-1'
        )

        try:
            # read the original table without binary genres
            movie_table = movie_table_config.read_table(
                self.dataset_dir,
                columns=movie_table_config.primary_key + movie_columns
            )
            # read the binary genres table
            genres_table = movie_table_config.read_table(
                self.dataset_dir,
                columns=MOVIE_GENRES
            )
        except FileNotFoundError:
            return None

        # drop and remove the empty column
        movie_columns.remove('empty')
        movie_table.drop('empty', axis=1, inplace=True)
        movie_table = movie_table[movie_table_config.primary_key + movie_columns]

        # replace 0 with NaN and 1 with the corresponding genre
        for column_name in genres_table:
            genres_table[column_name].replace({1:column_name, 0: np.nan}, inplace=True)

        # collapse genres into one column and add it to the original table
        genre_column = 'movie_genres'
        movie_table[genre_column] = genres_table.apply(lambda x: x.str.cat(sep='|'), axis=1)

        # update movie table definition
        movie_table_config.file.name = TABLE_FILE_PREFIX + self.dataset_name + '_movies.tsv.bz2'
        movie_table_config.file.options.compression = 'bz2'
        movie_table_config.file.options.sep = None
        movie_table_config.columns = movie_columns + [genre_column]
        movie_table_config.num_records = len(movie_table)

        # store the generated movie table
        movie_table_config.save_table(movie_table, self.dataset_dir)

        return movie_table_config

    def process_user_table(self) -> Optional[DatasetTableConfig]:
        """Process the user table.

        Changes the contents of the gender and occupation columns to be more user-friendly.

        Returns:
            the user table configuration or None on failure.
        """
        user_table_config = create_dataset_table_config(
            'u.user',
            ['user_id'],
            ['user_age', 'user_gender', 'user_occupation', 'user_zip code'],
            sep='|'
        )

        try:
            user_table = user_table_config.read_table(self.dataset_dir)
            user_table_config.num_records=len(user_table)
        except FileNotFoundError:
            return None

        # convert gender and occupation to more user-friendly names
        user_table['user_gender'].replace({'M': 'Male', 'F': 'Female'}, inplace=True)
        user_table['user_occupation'] = user_table['user_occupation'].str.capitalize()

        # update user table configuration
        user_table_config.file.name = TABLE_FILE_PREFIX + self.dataset_name + '_users.tsv.bz2'
        user_table_config.file.options.compression = 'bz2'
        user_table_config.file.options.sep = None

        # store the generated user table
        user_table_config.save_table(user_table, self.dataset_dir)

        return user_table_config
