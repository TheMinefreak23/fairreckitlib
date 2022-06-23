"""This module contains functionality to create matrices from dataset event tables.

Classes:

    MatrixProcessorConfig: the matrix processor configuration.
    DatasetMatrixProcessor: the dataset matrix processor that generates/adds user-item matrices.

Functions:

    create_matrix_chunk: create a user-item matrix chunk by counting user-item occurrences.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from ...core.events.event_dispatcher import EventDispatcher
from ...core.io.event_io import get_io_events, get_io_event_print_switch
from ...core.io.io_create import create_dir
from ...core.io.io_delete import delete_dir
from ...core.io.io_utility import save_yml
from ..ratings.convert_constants import RATING_TYPE_THRESHOLD
from .dataset import Dataset
from .dataset_constants import TABLE_FILE_PREFIX, DATASET_CONFIG_FILE
from .dataset_config import DATASET_RATINGS_EXPLICIT, DATASET_RATINGS_IMPLICIT, \
    DatasetTableConfig, DatasetMatrixConfig, DatasetIndexConfig, RatingMatrixConfig, \
    create_dataset_table_config

DEFAULT_MATRIX_CHUNK_SIZE = 10E6


@dataclass
class MatrixProcessorConfig:
    """Matrix Processor Configuration.

    event_table_name: the name of the event table to use.
    item_key: the item key name to create a user-item matrix for.
    rating_column: the name of the rating column to use in the user-item matrix.
    """

    event_table_name: str
    item_key: str
    rating_column: str


class DatasetMatrixProcessor:
    """Dataset Matrix Processor.

    The intended use of this class is to utilize an existing dataset that has event tables present,
    in order to generate a new user-item matrix. The processor does the following steps in order:

    1) create a temporary directory to store user-item chunks.
    2) process the event table by creating and saving user-item chunks.
    3) process the matrix by merging chunks.
    4) save the matrix in the dataset directory.
    5) update the dataset configuration file with the new user-item matrix.
    6) remove the temporary directory with.

    Public methods:

    run
    """

    def __init__(self, dataset: Dataset, verbose=True):
        """Construct the dataset matrix processor.

        Args:
            dataset: the dataset to create a new user-item matrix for.
            verbose: whether the processor should give verbose output.
        """
        self.dataset = dataset
        self.verbose = verbose
        print_event = lambda _, args: get_io_event_print_switch()[args.event_id](args)

        self.event_dispatcher = EventDispatcher()
        for event_id in get_io_events():
            self.event_dispatcher.add_listener(event_id, None, (print_event, None))

    def run(self,
            processor_config: MatrixProcessorConfig,
            *,
            chunk_size: int=DEFAULT_MATRIX_CHUNK_SIZE) -> bool:
        """Run the processor with the specified matrix configuration.

        The processor fails when the user-item matrix is already present in the
        existing dataset configuration.

        Args:
            processor_config: the configuration to use for creating a user-item matrix.
            chunk_size: the size of the chunks to use during processing.

        Raises:
            KeyError: when the event table does not exist in the dataset.
            IndexError: when the item key name is not present in the event table.

        Returns:
            whether the processing of the user-item matrix succeeded.
        """
        if not processor_config.event_table_name in self.dataset.config.events:
            raise KeyError('Event table does not exist: ' + processor_config.event_table_name)

        event_table = self.dataset.config.events[processor_config.event_table_name]
        item_key = processor_config.item_key
        if item_key not in event_table.primary_key:
            raise IndexError('Event table does not have the requested item key')

        matrix_name = 'user-' + item_key.split('_')[0] + '-' + processor_config.rating_column
        if self.dataset.get_matrix_config(matrix_name) is not None:
            return False

        # step 1
        temp_dir = create_dir(
            os.path.join(self.dataset.data_dir, TABLE_FILE_PREFIX + 'tmp'),
            self.event_dispatcher
        )

        if self.verbose:
            print('Started processing matrix')

        # step 2
        num_chunks = self.process_event_table(
            temp_dir,
            event_table,
            item_key,
            processor_config.rating_column,
            chunk_size=chunk_size
        )

        # step 3
        matrix_tuple = self.process_matrix_from_chunks(
            temp_dir,
            num_chunks,
            item_key,
            processor_config.rating_column
        )

        # step 4
        dataset_matrix_config = self.save_matrix(
            matrix_name,
            matrix_tuple,
            item_key,
            processor_config.rating_column
        )

        # step 5
        self.dataset.config.matrices[matrix_name] = dataset_matrix_config
        save_yml(
            os.path.join(self.dataset.data_dir, DATASET_CONFIG_FILE),
            self.dataset.config.to_yml_format()
        )

        if self.verbose:
            print('Finished processing matrix')

        # step 6
        delete_dir(temp_dir, self.event_dispatcher)
        return True

    def process_event_table(
            self,
            output_dir: str,
            event_table: DatasetTableConfig,
            item_key: str,
            rating_column: str,
            *,
            chunk_size: int=DEFAULT_MATRIX_CHUNK_SIZE) -> int:
        """Process the event table in chunks.

        Args:
            output_dir: the output directory to store the chunks.
            event_table: the event table to process into chunks.
            item_key: the item key name to create a user-item chunk for.
            rating_column: the name of the rating column to use in the user-item chunk.
            chunk_size: the size of the chunks to use during processing.

        Returns:
            the number of chunks that are generated.
        """
        if self.verbose:
            print('Started processing event table')

        num_chunks = 0
        start_row = 0
        event_table_it = event_table.read_table(self.dataset.data_dir, chunk_size=chunk_size)
        for i, chunk in enumerate(event_table_it):
            end_row = int(min(start_row + chunk_size, event_table.num_records))
            percent = float(start_row) / float(event_table.num_records) * 100.0
            if self.verbose:
                print('Processing rows', start_row, 'to', end_row,
                      'of', event_table.num_records, '=>', str(percent) + '%')

            chunk = create_matrix_chunk(chunk, item_key, rating_column)
            # TODO filter chunk based on user/item columns
            chunk.to_csv(os.path.join(output_dir, 'chunk_' + str(i) + '.tsv'),
                         sep='\t', header=True, index=False)

            num_chunks += 1
            start_row += chunk_size

        if self.verbose:
            print('Finished processing event table')

        return num_chunks

    def process_matrix_from_chunks(
            self,
            output_dir: str,
            num_chunks: int,
            item_key: str,
            rating_column: str) -> Tuple[pd.DataFrame, List[int], List[int]]:
        """Process the user-item matrix from the stored chunks.

        Args:
            output_dir: the output directory where the chunks are stored.
            num_chunks: the number of chunks in the output directory.
            item_key: the item key name that was used to create user-item chunks.
            rating_column: the name of the rating column that was used to create user-item chunks.

        Returns:
            the user-item matrix and the lists of unique user/item ids.
        """
        if self.verbose:
            print('Started processing matrix chunks')

        matrix = pd.DataFrame({
            'user_id': pd.Series(dtype='int'),
            item_key: pd.Series(dtype='int'),
            rating_column: pd.Series(dtype='float')
        })

        for i in range(num_chunks):
            if self.verbose:
                print('Processing matrix chunk', i + 1, 'of', num_chunks)

            chunk = pd.read_table(os.path.join(output_dir, 'chunk_' + str(i) + '.tsv'))
            matrix = pd.concat([matrix, chunk], ignore_index=True)
            matrix = matrix.groupby(['user_id', item_key], as_index=False).sum()

        unique_users = matrix['user_id'].unique()
        # users from 0...num_users
        matrix = pd.merge(
            matrix,
            pd.DataFrame(list(enumerate(unique_users)), columns=['user', 'user_id']),
            how='left', on='user_id'
        )
        matrix.drop('user_id', axis=1, inplace=True)

        unique_items = matrix[item_key].unique()
        # items from 0...num_items
        matrix = pd.merge(
            matrix,
            pd.DataFrame(list(enumerate(unique_items)), columns=['item', item_key]),
            how='left', on=item_key
        )
        matrix.drop(item_key, axis=1, inplace=True)

        if self.verbose:
            print('Finished processing matrix chunks')

        return matrix[['user', 'item', rating_column]], list(unique_users), list(unique_items)

    def save_matrix(
            self,
            matrix_name: str,
            matrix_tuple: Tuple[pd.DataFrame, List[int], List[int]],
            item_key: str,
            rating_column: str) -> DatasetMatrixConfig:
        """Save the matrix in the dataset directory.

        Args:
            matrix_name: the name that is used to save the matrix, user and item lists.
            matrix_tuple: the user-item matrix and the lists of unique user/item ids.
            item_key: the item key name that was used to create the user-item matrix.
            rating_column: name of the rating column that was used to create the user-item matrix.

        Returns:
            the dataset matrix configuration.
        """
        if self.verbose:
            print('Started saving matrix')

        dataset_matrix_name = TABLE_FILE_PREFIX + self.dataset.get_name() + '_' + matrix_name
        matrix, users, items = matrix_tuple

        # create the user indices config and save the array
        user_index_config = DatasetIndexConfig(
            dataset_matrix_name + '_user_indices.hdf5',
            'user_id',
            len(users)
        )
        user_index_config.save_indices(self.dataset.data_dir, users)

        # create the item indices config and save the array
        item_index_config = DatasetIndexConfig(
            dataset_matrix_name + '_item_indices.hdf5',
            item_key,
            len(items)
        )
        item_index_config.save_indices(self.dataset.data_dir, items)

        # create the sample matrix table config and save the table
        matrix_table_config = create_dataset_table_config(
            dataset_matrix_name + '_matrix.tsv.bz2',
            ['user_id', item_key],
            ['matrix_' + rating_column],
            compression='bz2',
            foreign_keys=['user_id', item_key],
            num_records=len(matrix)
        )
        matrix_table_config.save_table(matrix, self.dataset.data_dir)

        rating_min = float(matrix[rating_column].min())
        rating_max = float(matrix[rating_column].max())
        rating_type = DATASET_RATINGS_IMPLICIT \
            if rating_max > RATING_TYPE_THRESHOLD else DATASET_RATINGS_EXPLICIT

        if self.verbose:
            print('Finished saving matrix')

        return DatasetMatrixConfig(
            matrix_table_config,
            RatingMatrixConfig(
                rating_min,
                rating_max,
                rating_type
            ),
            user_index_config,
            item_index_config
        )



def create_matrix_chunk(chunk: pd.DataFrame, item_key: str, rating_column: str) -> pd.DataFrame:
    """Create a user-item matrix chunk by counting occurrences of user-item combinations.

    Args:
        chunk: a dataframe chunk containing the user-item events.
        item_key: the key of the item to use for counting occurrences.
        rating_column: the name of the rating column to store the user-item counter in.

    Returns:
        a dataframe with the columns ['user', item_key, rating_column].
    """
    # make a copy to prevent pandas slicing errors and drop any irrelevant columns
    chunk_header = ['user_id', item_key]
    chunk = pd.DataFrame(chunk[chunk_header])

    chunk[rating_column] = np.ones(len(chunk))
    chunk = chunk.groupby(chunk_header, as_index=False).count()
    return chunk
