"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import os

from ...utility import save_array_to_hdf5
from ...utility import save_yml
from ..dataset_config import DATASET_INDICES
from ..dataset_config import DATASET_ITEMS
from ..dataset_config import DATASET_MATRIX
from ..dataset_config import DATASET_PREFIX
from ..dataset_config import DATASET_TABLES
from ..dataset_config import DATASET_USERS
from ..dataset_table import DATASET_FILE
from ..dataset_table import read_table, write_table


class DataProcessorBase(metaclass=ABCMeta):
    """DataProcessor base class for all FairRecKit datasets.

    Datasets are preprocessed so that they will be of a recognized standard format
    on the other side. A configuration file is produced in the resulting dataset
    directory that stores the metadata for achieving this. For further information
    it is advised to take a look at the Dataset class.

    Args:
        dataset_name(str): name of the dataset (processor).
        matrix_file(str): name of the dataset matrix file.
    """
    def __init__(self, dataset_name, matrix_file):
        self.dataset_name = dataset_name
        self.matrix_file = matrix_file

        # data buffers that are accessible during processing
        self.data_matrix = None
        self.user_list = None
        self.item_list = None

    @abstractmethod
    def load_matrix(self, file_path):
        """Loads the matrix of the dataset.

        The matrix is expected to be of the desired standardized format,
        meaning a pandas.DataFrame as described in the Dataset class.
        The dataset frame should be stored in the self.data_matrix before returning.

        When the dataset requires indirection arrays for the user and/or item IDs, they
        should be stored in the self.user_list or self.item_list respectively before returning.

        Arg:
            file_path(str): the total path to the matrix file.

        Returns:
            save_matrix(bool): whether the processed matrix needs to be stored.
            rating_type(str): the rating type of the matrix, either 'explicit' or 'implicit'.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_tables_config(self, dataset_dir, user_item_tables):
        """Loads any other tables available for the dataset.

        Args:
            dataset_dir(str): directory where the dataset files are present.
            user_item_tables(dict): dictionary containing the loaded user
                and item table configurations.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_user_table_config(self, dataset_dir):
        """Loads the user table of the dataset.

        Args:
            dataset_dir(str): directory where the dataset files are present.

        Returns:
            user_table_name(str): name of the user table or None when not available.
            user_table_config(dict): user table configuration or None when not available.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_item_table_config(self, dataset_dir):
        """Loads the item table of the dataset.

        Args:
            dataset_dir(str): directory where the dataset files are present.

        Returns:
            item_table_name(str): name of the item table or None when not available.
            item_table_config(dict): item table configuration or None when not available.
        """
        raise NotImplementedError()

    def load_user_indices(self, dataset_dir):
        """Loads the user indices of the dataset.

        This step happens at the end of processing and will save
        the self.user_list, when still present, to the dataset directory.

        Args:
            dataset_dir(str): directory where the dataset files are present.

        Returns:
            file_name(str): the name to the user indices file or None when not present.
            user_count(int): the total number of unique users.
        """
        file_name = None
        if not self.user_list is None:
            user_count = len(self.user_list)
            file_name = DATASET_PREFIX + self.dataset_name + '_idx_users.hdf5'

            save_array_to_hdf5(
                os.path.join(dataset_dir, file_name),
                self.user_list,
                DATASET_INDICES
            )
        else:
            user_count = len(self.data_matrix['user'].unique())

        return file_name, user_count

    def load_item_indices(self, dataset_dir):
        """Loads the item indices of the dataset.

        This step happens at the end of processing and will save
        the self.item_list, when still present, to the dataset directory.

        Args:
            dataset_dir(str): directory where the dataset files are present.

        Returns:
            file_name(str): the name to the item indices file or None when not present.
            item_count(int): the total number of unique items.
        """
        file_name = None
        if not self.item_list is None:
            item_count = len(self.item_list)
            file_name = DATASET_PREFIX + self.dataset_name + '_idx_items.hdf5'

            save_array_to_hdf5(
                os.path.join(dataset_dir, file_name),
                self.item_list,
                DATASET_INDICES
            )
        else:
            item_count = len(self.data_matrix['item'].unique())

        return file_name, item_count

    def process_matrix(self, dataset_dir):
        """Processes the matrix of the dataset.

        Loads the matrix and stores it in the dataset directory when needed.

        Args:
            dataset_dir(str): directory where the dataset files are present.

        Returns:
            rating_type(str): the rating type of the matrix, either 'explicit' or 'implicit'.
        """
        save_matrix, rating_type = self.load_matrix(os.path.join(dataset_dir, self.matrix_file))

        # save to disk when not in the desired format and update matrix file name
        if save_matrix:
            self.matrix_file = DATASET_PREFIX + self.dataset_name + '_matrix.tsv'
            write_table(
                self.data_matrix,
                dataset_dir,
                self.matrix_file
            )

        return rating_type

    def process_tables(self, dataset_dir):
        """Processes the matrix of the dataset.

        Loads all the tables belonging to the dataset:

        1) load user table
        2) load item table
        3) load other tables

        Args:
            dataset_dir(str): directory where the dataset files are present.

        Returns:
            user_table_name(str): name of the user table or None when not present.
            item_table_name(str): name of the item table or None when not present.
            tables_config(dict): dictionary containing the total configuration of all
                tables. The key is the name of the table, and the value is the configuration.
        """
        # attempt to load the user table
        try:
            user_table_name, user_table = self.load_user_table_config(dataset_dir)
        except FileNotFoundError:
            print('Processor', self.dataset_name, 'failed to load user table!')
            user_table_name, user_table = None, None

        # attempt to load the item table
        try:
            item_table_name, item_table = self.load_item_table_config(dataset_dir)
        except FileNotFoundError:
            print('Processor', self.dataset_name, 'failed to load item table!')
            item_table_name, item_table = None, None

        tables_config = {}
        if user_table_name and user_table:
            tables_config[user_table_name] = user_table
        if item_table_name and item_table:
            tables_config[item_table_name] = item_table

        # attempt to load other tables
        try:
            tables_config = self.load_tables_config(dataset_dir, tables_config)
        except FileNotFoundError:
            print('Processor', self.dataset_name, 'failed to load other tables!')

        # validate table configurations
        for table_name, config in dict(tables_config).items():
            try:
                # read table in small chunks in case there are very large tables
                table_iterator = read_table(
                    dataset_dir,
                    config,
                    config['keys'],
                    chunk_size=10000000
                )

                tables_config[table_name]['num_records'] = 0
                for _, table in enumerate(table_iterator):
                    tables_config[table_name]['num_records'] += len(table)

            except FileNotFoundError:
                del tables_config[table_name]

        return user_table_name, item_table_name, tables_config

    def run(self, dataset_dir, config_file_name):
        """Runs the processor over a stored dataset.

        Args:
            dataset_dir(str): directory where the dataset files are present.
            config_file_name(str): name of the file to save the dataset
                configuration to.

        Returns:
            dataset_config(dict): the configuration of the dataset after processing
                or None on failure.
        """
        # attempt to process the matrix file
        try:
            rating_type = self.process_matrix(dataset_dir)
        except FileNotFoundError:
            return None

        user_table_name, item_table_name, tables_config = self.process_tables(dataset_dir)

        # load relevant data from the user/item indices
        user_file_name, user_count = self.load_user_indices(dataset_dir)
        item_file_name, item_count = self.load_item_indices(dataset_dir)

        dataset_config = {
            DATASET_MATRIX: {
                DATASET_FILE: self.matrix_file,
                'num_pairs': len(self.data_matrix),
                'num_users': user_count,
                'num_items': item_count,
                'rating_min': float(self.data_matrix['rating'].min()),
                'rating_max': float(self.data_matrix['rating'].max()),
                'rating_type': rating_type,
                'timestamp': self.data_matrix.get('timestamp') is not None
            },
            DATASET_USERS: {DATASET_FILE: user_file_name, 'table': user_table_name},
            DATASET_ITEMS: {DATASET_FILE: item_file_name, 'table': item_table_name},
            DATASET_TABLES: tables_config
        }

        save_yml(
            os.path.join(dataset_dir, config_file_name),
            dataset_config
        )

        return dataset_config
