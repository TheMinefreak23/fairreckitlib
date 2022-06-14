"""This module contains the base processor for LastFM datasets.

Classes:

    DatasetProcessorLFM: the base class for LastFM dataset processors.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..dataset_config import DATASET_RATINGS_IMPLICIT, RatingMatrixConfig
from ..dataset_config import DatasetIndexConfig, DatasetMatrixConfig, DatasetTableConfig
from ..dataset_constants import TABLE_FILE_PREFIX
from .dataset_processor_base import DatasetProcessorBase


class DatasetProcessorLFM(DatasetProcessorBase, metaclass=ABCMeta):
    """DataProcessor base class for LastFM datasets.

    Provides an abstraction for processing the listening event table,
    and also for generalizing the user table data. An iterative matrix
    processor function is exposed for derived subclasses as the LastFM
    dataset matrices tend to be very big.

    Abstract methods:

    create_listening_events_config
    create_user_table_config
    """

    @abstractmethod
    def create_listening_events_config(self) -> Optional[DatasetTableConfig]:
        """Create the listening event table configuration.

        Returns:
            the configuration of the listening event table or None when not available.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_user_table_config(self) -> DatasetTableConfig:
        """Create the user table configuration.

        Returns:
            the configuration of the user table.
        """
        raise NotImplementedError()

    def get_event_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get event table configuration processors.

        Returns:
            a list containing the listening event table processor.
        """
        return [('listening event', self.process_listening_events)]

    def get_table_configs(self) -> List[Tuple[str, Callable[[], Optional[DatasetTableConfig]]]]:
        """Get table configuration processors.

        Derived implementations are expected to call the super implementation in
        order to include the user table in the configuration.

        Returns:
            a list containing the user table processor.
        """
        return [('user', self.process_user_table)]

    def process_listening_events(self) -> Optional[DatasetTableConfig]:
        """Process the listening event table.

        Returns:
            the listening event table configuration or None on failure.
        """
        les_table_config = self.create_listening_events_config()
        # skip without table configuration
        if les_table_config is None:
            return None

        try:
            # count records in chunks as these files are huge
            table_iterator = les_table_config.read_table(self.dataset_dir, chunk_size=50000000)
            for _, table in enumerate(table_iterator):
                les_table_config.num_records += len(table)

            return les_table_config
        except FileNotFoundError:
            return None

    def process_matrix(
            self,
            matrix_table_config: DatasetTableConfig,
            user_idx_file: str=None,
            item_idx_file: str=None) -> Optional[DatasetMatrixConfig]:
        """Process the matrix with the specified configuration.

        Args:
            matrix_table_config: the configuration of the matrix to process.
            user_idx_file: the file name of the user indices or None when not present.
            item_idx_file: the file name of the item indices or None when not present.

        Returns:
            the matrix configuration or None on failure.
        """
        user_id = matrix_table_config.primary_key[0]
        item_id = matrix_table_config.primary_key[1]
        count_column = matrix_table_config.columns[0]

        unique_users = []
        unique_items = []
        rating_min = 1000000000.0
        rating_max = 0.0

        try:
            matrix_it = matrix_table_config.read_table(self.dataset_dir, chunk_size=50000000)
            # process matrix in chunks
            for _, matrix in enumerate(matrix_it):
                unique_users = pd.Series(unique_users).append(matrix[user_id]).unique()
                unique_items = pd.Series(unique_items).append(matrix[item_id]).unique()
                matrix_table_config.num_records += len(matrix)
                rating_min = min(rating_min, matrix[count_column].min())
                rating_max = max(rating_max, matrix[count_column].max())
        except FileNotFoundError:
            return None

        return DatasetMatrixConfig(
            matrix_table_config,
            RatingMatrixConfig(
                rating_min,
                rating_max,
                DATASET_RATINGS_IMPLICIT
            ),
            DatasetIndexConfig(
                user_idx_file,
                user_id,
                len(unique_users)
            ),
            DatasetIndexConfig(
                item_idx_file,
                item_id,
                len(unique_items)
            )
        )

    def process_user_table(self) -> Optional[DatasetTableConfig]:
        """Process the user table.

        Changes the contents of the gender column to be more user-friendly,
        and the contents of the age column to -1 when above 100.

        Returns:
            the user table configuration or None on failure.
        """
        user_table_config = self.create_user_table_config()

        try:
            user_table = user_table_config.read_table(self.dataset_dir)
            user_table_config.num_records = len(user_table)
        except FileNotFoundError:
            return None

        # convert gender to more user-friendly names
        user_table['user_gender'].replace({
            'm': 'Male',
            'f': 'Female',
            'n': 'Neutral'
        }, inplace=True)
        # convert age above 100 to -1
        user_table['user_age'].mask(user_table['user_age'].gt(100), inplace=True)
        user_table['user_age'].fillna(-1.0, inplace=True)
        user_table['user_age'] = user_table['user_age'].astype(int)

        # update table configuration
        user_table_config.file.name = TABLE_FILE_PREFIX + self.dataset_name + '_users.tsv.bz2'
        user_table_config.file.options.compression = 'bz2'
        user_table_config.file.options.header = False
        user_table_config.file.options.sep = None

        # store the resulting user table
        user_table_config.save_table(user_table, self.dataset_dir)

        return user_table_config
