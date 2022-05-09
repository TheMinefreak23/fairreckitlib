"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from ..config import DATASET_FILE
from ..config import DATASET_PREFIX
from ..config import DATASET_RATINGS_EXPLICIT
from ..table import create_table_config
from ..table import read_table
from ..table import write_table
from ..utility import load_table
from .processor_base import DataProcessorBase


class DataProcessorML25M(DataProcessorBase):
    """DataProcessor for the ML-25M dataset.

    Args:
        dataset_name(str): name of the dataset.
    """
    def __init__(self, dataset_name):
        DataProcessorBase.__init__(self, dataset_name, 'ratings.csv')

    def load_matrix(self, file_path):
        self.data_matrix = load_table(
            file_path,
            ['user', 'item', 'rating', 'timestamp'],
            sep=',',
            header=True
        )

        return True, DATASET_RATINGS_EXPLICIT

    def load_tables_config(self, dataset_dir, user_item_tables):
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

    def load_user_table_config(self, dataset_dir):
        return None, None

    def load_item_table_config(self, dataset_dir):
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

        movie_table = pd.merge(movie_table, link_table, how='left', on='movie_id')

        movie_table_config[DATASET_FILE] = DATASET_PREFIX + self.dataset_name + '_movies.tsv'
        movie_table_config['sep']=None
        movie_table_config['columns'] += link_table_config['columns']

        write_table(
            movie_table,
            dataset_dir,
            movie_table_config[DATASET_FILE]
        )

        return 'movie', movie_table_config
