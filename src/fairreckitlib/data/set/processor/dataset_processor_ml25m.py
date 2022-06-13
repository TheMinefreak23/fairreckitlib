"""This modules contains the class to process the MovieLens-25M dataset.

Classes:

    DatasetProcessorML25M: data processor implementation for the ML-25M dataset.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..dataset_config import DatasetTableConfig
from ..dataset_config import create_dataset_table_config
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_ml import DatasetProcessorML


class DatasetProcessorML25M(DatasetProcessorML):
    """DatasetProcessor for the MovieLens-25M dataset.

    The dataset can be downloaded from the link below.
    https://files.grouplens.org/datasets/movielens/ml-25m.zip

    The processor handles the following files:

    genome-scores.csv (optional)
    genome-tags.csv (optional)
    links.csv (optional)
    movies.csv (optional)
    ratings.csv (required)
    tags.csv (optional)
    """

    def create_user_movie_matrix_config(self) -> DatasetTableConfig:
        """Create the user-movie matrix configuration.

        Returns:
            the table configuration of the ML-25M matrix.
        """
        return create_dataset_table_config(
            'ratings.csv',
            ['user_id', 'movie_id'],
            ['matrix_rating', 'matrix_timestamp'],
            foreign_keys=['user_id', 'movie_id'],
            header=True,
            sep=','
        )

    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Returns:
            a list containing the genome score, genome tag, movie and tag table processors.
        """
        return [
            ('genome score', self.process_genome_score_table),
            ('genome tag', self.process_genome_tag_table),
            ('movie', self.process_movie_table),
            ('tag', self.process_tag_table)
        ]

    def process_genome_score_table(self) -> Optional[DatasetTableConfig]:
        """Process the genome score table.

        Returns:
            the genome score table configuration or None on failure.
        """
        genome_score_table_config = create_dataset_table_config(
            'genome-scores.csv',
            ['movie_id', 'tag_id'],
            ['relevance'],
            foreign_keys=['movie_id', 'tag_id'],
            header=True,
            sep=','
        )

        try:
            genome_score_table = genome_score_table_config.read_table(self.dataset_dir)
            genome_score_table_config.num_records = len(genome_score_table)
            return genome_score_table_config
        except FileNotFoundError:
            return None

    def process_genome_tag_table(self) -> Optional[DatasetTableConfig]:
        """Process the genome tag table.

        Returns:
            the genome tag table configuration or None on failure.
        """
        genome_tag_table_config = create_dataset_table_config(
            'genome-tags.csv',
            ['tag_id'],
            ['tag_name'],
            header=True,
            sep=','
        )

        try:
            genome_tag_table = genome_tag_table_config.read_table(self.dataset_dir)
            genome_tag_table_config.num_records = len(genome_tag_table)
            return genome_tag_table_config
        except FileNotFoundError:
            return None

    def process_movie_table(self) -> Optional[DatasetTableConfig]:
        """Process the movie table.

        The movie and link tables are joined together for simplification.

        Returns:
            the movie table configuration or None on failure.
        """
        link_table_config = create_dataset_table_config(
            'links.csv',
            ['movie_id'],
            ['movie_imdbID', 'movie_tmdbID'],
            header=True,
            sep=','
        )

        try:
            link_table = link_table_config.read_table(self.dataset_dir)
            link_table_config.num_records = len(link_table)
            # replace NaN and cast back to original int
            link_table['movie_tmdbID'] = link_table['movie_tmdbID'].fillna(-1.0).astype(int)
        except FileNotFoundError:
            link_table = None

        movie_table_config = create_dataset_table_config(
            'movies.csv',
            ['movie_id'],
            ['movie_title', 'movie_genres'],
            header=True,
            sep=','
        )

        try:
            movie_table = movie_table_config.read_table(self.dataset_dir)
            movie_table_config.num_records = len(movie_table)
        except FileNotFoundError:
            return link_table_config if link_table is not None else None

        if link_table is not None:
            # merge movie and link tables
            movie_table = pd.merge(movie_table, link_table, how='left', on='movie_id')

            # update movie table configuration
            movie_table_config.file.name = TABLE_FILE_PREFIX + self.dataset_name + '_movies.tsv.bz2'
            movie_table_config.file.options.sep = None
            movie_table_config.file.options.compression = 'bz2'
            movie_table_config.file.options.header = False
            movie_table_config.columns += link_table_config.columns

            # store the extended movie table
            movie_table_config.save_table(movie_table, self.dataset_dir)

        return movie_table_config

    def process_tag_table(self) -> Optional[DatasetTableConfig]:
        """Process the tag table.

        Returns:
            the tag table configuration or None on failure.
        """
        tag_table_config = create_dataset_table_config(
            'tags.csv',
            ['user_id', 'movie_id'],
            ['tag_name','tag_timestamp'],
            foreign_keys=['user_id', 'movie_id'],
            header=True,
            sep=','
        )

        try:
            tag_table = tag_table_config.read_table(self.dataset_dir)
            tag_table_config.num_records = len(tag_table)
            return tag_table_config
        except FileNotFoundError:
            return None
