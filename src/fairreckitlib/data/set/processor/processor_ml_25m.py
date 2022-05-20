"""This modules contains the class to process the MovieLens-25M dataset.

Classes:

    DataProcessorML25M: data processor implementation for the ML-25M dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, Tuple

import pandas as pd

from ...utility import load_table
from ..dataset import DATASET_RATINGS_EXPLICIT
from ..dataset_constants import DATASET_FILE, DATASET_PREFIX
from ..dataset_table import create_table_config, read_table, write_table
from .processor_base import DataProcessorBase


class DataProcessorML25M(DataProcessorBase):
    """DataProcessor for the MovieLens-25M dataset."""

    def __init__(self, dataset_name):
        """Construct the ML-25M DataProcessor.

        Args:
            dataset_name: name of the dataset.
        """
        DataProcessorBase.__init__(self, dataset_name, 'ratings.csv')

    def load_matrix(self, file_path: str) -> Tuple[bool, str]:
        """Load the matrix of the dataset.

        The user-item matrix is stored in a csv file.
        No additional processing is needed.

        Args:
            file_path: the path to the matrix file.

        Returns:
            True, DATASET_RATINGS_EXPLICIT
        """
        self.data_matrix = load_table(
            file_path,
            ['user', 'item', 'rating', 'timestamp'],
            sep=',',
            header=True
        )

        return True, DATASET_RATINGS_EXPLICIT

    def load_tables_config(self, dataset_dir: str, user_item_tables: Dict[str, Dict[str, Any]]):
        """Load the genome scores, genome tags and tag tables of the dataset.

        Args:
            dataset_dir: directory where the dataset files are present.
            user_item_tables: dictionary containing the loaded user
                and artist table configurations.

        Returns:
            a dictionary with user, movie, genome scores, genome tags and tag configurations.
        """
        genome_score_table_config = create_table_config(
            'genome-scores.csv',
            ['movie_id', 'tag_id'],
            ['relevance'],
            header=True,
            sep=','
        )

        genome_tag_table_config = create_table_config(
            'genome-tags.csv',
            ['tag_id'],
            ['tag_name'],
            header=True,
            sep=','
        )

        tag_table_config = create_table_config(
            'tags.csv',
            ['user_id', 'movie_id'],
            ['tag_name','tag_timestamp'],
            header=True,
            sep=','
        )

        user_item_tables['genome_score'] = genome_score_table_config
        user_item_tables['genome_tag'] = genome_tag_table_config
        user_item_tables['tag'] = tag_table_config

        return user_item_tables

    def load_user_table_config(self, dataset_dir: str) -> Tuple[None, None]:
        """Return nothing, no user table is available."""
        return None, None

    def load_item_table_config(self, dataset_dir: str) -> Tuple[str, Dict[str, Any]]:
        """Load the movie table of the dataset.

        The movie and link tables are joined together for simplification.

        Args:
            dataset_dir: directory where the dataset files are present.

        Returns:
            the name and configuration of the movie table.
        """
        link_table_config = create_table_config(
            'links.csv',
            ['movie_id'],
            ['movie_imdbID', 'movie_tmdbID'],
            header=True,
            sep=','
        )

        link_table = read_table(
            dataset_dir,
            link_table_config
        )

        # replace NaN and cast back to original int
        link_table['movie_tmdbID'] = link_table['movie_tmdbID'].fillna(-1.0).astype(int)

        movie_table_config = create_table_config(
            'movies.csv',
            ['movie_id'],
            ['movie_title', 'movie_genres'],
            header=True,
            sep=','
        )

        movie_table = read_table(
            dataset_dir,
            movie_table_config
        )

        # merge movie and link tables
        movie_table = pd.merge(movie_table, link_table, how='left', on='movie_id')

        # update movie table configuration
        movie_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_movies.tsv'
        movie_table_config['sep']=None
        movie_table_config['columns'] += link_table_config['columns']

        # store the extended movie table
        write_table(
            movie_table,
            dataset_dir,
            movie_table_config[DATASET_FILE]
        )

        return 'movie', movie_table_config
