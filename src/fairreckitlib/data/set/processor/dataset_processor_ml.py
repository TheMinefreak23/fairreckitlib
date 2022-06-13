"""This module contains the base processor for MovieLens datasets.

Classes:

    DatasetProcessorML: the base class for MovieLens dataset processors.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..dataset_config import DATASET_RATINGS_EXPLICIT, RatingMatrixConfig
from ..dataset_config import DatasetIndexConfig, DatasetMatrixConfig, DatasetTableConfig
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_base import DatasetProcessorBase


class DatasetProcessorML(DatasetProcessorBase, metaclass=ABCMeta):
    """DataProcessor base class for MovieLens datasets.

    Provides an abstraction for processing the user-movie-rating matrix.
    Moreover, it is assumed that the datasets do not have any event tables.

    Abstract methods:

    create_user_movie_matrix_config
    """

    @abstractmethod
    def create_user_movie_matrix_config(self) -> DatasetTableConfig:
        """Create the user-movie matrix configuration.

        Returns:
            the table configuration of the matrix.
        """
        raise NotImplementedError()

    def get_event_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get event table configuration processors.

        Returns:
            an empty list.
        """
        return []

    def get_matrix_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetMatrixConfig]]]]:
        """Get matrix configuration processors.

        Returns:
            a list with the user-movie-rating matrix processor.
        """
        return [('user-movie-rating', self.process_user_movie_matrix)]

    def process_user_movie_matrix(self) -> Optional[DatasetMatrixConfig]:
        """Process the user-movie-rating matrix.

        Returns:
            the matrix configuration or None on failure.
        """
        user_movie_matrix_table_config = self.create_user_movie_matrix_config()

        # extract column names from configuration
        user_id = user_movie_matrix_table_config.primary_key[0]
        item_id = user_movie_matrix_table_config.primary_key[1]
        rating_column = user_movie_matrix_table_config.columns[0]

        try:
            user_movie_matrix = user_movie_matrix_table_config.read_table(self.dataset_dir)
            user_movie_matrix_table_config.num_records = len(user_movie_matrix)
        except FileNotFoundError:
            return None

        if user_movie_matrix[rating_column].dtype == np.int64:
            # convert int ratings
            user_movie_matrix[rating_column] = user_movie_matrix[rating_column].astype(float)

            # update matrix configuration
            user_movie_matrix_table_config.file.name = \
                TABLE_FILE_PREFIX + self.dataset_name + '_user-movie-rating_matrix.tsv.bz2'
            user_movie_matrix_table_config.file.options.sep = None
            user_movie_matrix_table_config.file.options.compression = 'bz2'
            user_movie_matrix_table_config.file.options.header = False

            # store resulting matrix
            user_movie_matrix_table_config.save_table(user_movie_matrix, self.dataset_dir)

        return DatasetMatrixConfig(
            user_movie_matrix_table_config,
            RatingMatrixConfig(
                user_movie_matrix[rating_column].min(),
                user_movie_matrix[rating_column].max(),
                DATASET_RATINGS_EXPLICIT
            ),
            DatasetIndexConfig(
                None,
                user_id,
                len(user_movie_matrix[user_id].unique())
            ),
            DatasetIndexConfig(
                None,
                item_id,
                len(user_movie_matrix[item_id].unique())
            )
        )
