"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pandas as pd

from .config import DATASET_FILE
from .config import DATASET_INDICES
from .config import DATASET_ITEMS
from .config import DATASET_MATRIX
from .config import DATASET_TABLES
from .config import DATASET_USERS
from .table import read_table
from .utility import load_array_from_hdf5


class Dataset:
    """Wrapper class for a FairRecKit dataset.

    A dataset is used for carrying out recommender system experiments.
    Each dataset has a strong affinity with a database structure consisting of
    multiple tables.
    The standardized matrix is a pandas.DataFrame stored in a '.tsv' file.
    The (derived sparse) matrix is used in experiments and needs to be
    in a CSR compatible format, meaning three fields:

    1) 'user': IDs range from 0 to the amount of unique users.
    2) 'item': IDs range from 0 to the amount of unique items. An item can be
        various of things (e.g. an artist, an album, a track, a movie, etc.)
    3) 'rating': floating-point data describing the rating a user has given an item.
        There are two types of ratings, namely explicit or implicit, and both
        are expected to be greater than zero.

    The matrix has one optional field which is:

    4) 'timestamp': when present can be used to split the matrix on temporal basis.

    A dataset has two main tables that are connected to the 'user' and 'item' fields.
    Indirection arrays are available when user and/or item IDs do not match up in
    their corresponding tables. These two tables can be used in an experiment to
    filter any rows based on various table header criteria.
    Any additional tables can be added for accessibility/compatibility with the FRK
    recommender system.

    Args:
        name(str): name of the dataset.
        data_dir(str): directory where the dataset is stored.
        config(dict): data configuration dictionary.
    """
    def __init__(self, name, data_dir, config):
        self.name = name
        self.data_dir = data_dir
        self.config = config

    def get_available_tables(self):
        """Gets the available table names in the dataset.

        Returns:
            table_names(array like): list of table names.
        """
        table_names = []

        for table_name, _ in self.config[DATASET_TABLES].items():
            table_names.append(table_name)

        return table_names

    def get_available_user_item_columns(self):
        """Gets the available user/item table column names of this dataset.

        The resulting dictionary is split into three categories:

        1) 'user_item': the column names related to a user-item pair, consisting of
            at least a rating and optionally a timestamp.
        2) 'user': the column names related to the user table or None when not available.
        3) 'item': the column names related to the item table or None when not available.

        Returns:
            (dict): containing for each category as the key another dict with:
                'name': the name of the category.
                'columns': the available columns for the category.

        """
        user_item_columns = ['rating']
        if self.get_user_table_info('timestamp'):
            user_item_columns.append('timestamp')

        result = {
            'user_item': {
                'name': 'matrix',
                'columns': user_item_columns
            }
        }

        user_table_name = self.get_user_table_name()
        if user_table_name is not None:
            result['user'] = {
                'name': user_table_name,
                'columns': self.get_table_info(user_table_name, 'columns')
            }

        item_table_name = self.get_item_table_name()
        if item_table_name is not None:
            result['item'] = {
                'name': item_table_name,
                'columns': self.get_table_info(item_table_name, 'columns')
            }

        return result

    def get_matrix_file_path(self):
        """Gets the file path where the matrix is stored.

        Returns:
            (str): the path of the dataset matrix file.
        """
        return os.path.join(
            self.data_dir,
            self.get_matrix_info(DATASET_FILE)
        )

    def get_matrix_info(self, key=None):
        """Gets information of the dataset matrix.

        The total dictionary that is available includes:

        file(str): the name of the matrix file including extension.
        num_pairs(int): total number of unique user and item combinations.
        num_users(int): total number of unique users.
        num_items(int): total number of unique items.
        rating_min(float): the minimum included rating.
        rating_max(float): the maximum included rating.
        rating_type(str): either 'explicit' or 'implicit.
        timestamp(bool): whether a timestamp is available.

        Args:
            key(str): the specific matrix information or None for all.

        Returns:
            (Any): the corresponding value to the key when specified or
                a dictionary with all information. When the key is not
                found None is returned.
        """
        if key is not None:
            return self.config[DATASET_MATRIX].get(key)

        return dict(self.config[DATASET_MATRIX])

    def get_table_info(self, table_name, key=None):
        """Gets information of the dataset table with the specified name.

        The total dictionary that is available includes:

        file(str): the name of the table file including extension.
        keys(array like): list of strings that make up the keys of the table.
            Concatenated before the 'columns' to get all available names.
        columns(array like): list of strings that make up the other available
            columns in the table. Concatenated after the 'keys' to get all available names.
        encoding(str): the encoding of the table contents.
        header(bool): whether the table contains a header on the first line.
        num_records(int): total number of rows in the table.
        sep(str): the delimiter that is used in the table or None for a tab separator.

        Args:
            table_name(str): name of the table to retrieve information from.
            key(str): the specific table information or None for all.

        Returns:
            (Any): the corresponding value to the key when specified or
                a dictionary with all information. When the key or table is not
                found None is returned.
        """
        if not table_name or table_name not in self.config[DATASET_TABLES]:
            return None

        if key is not None:
            return self.config[DATASET_TABLES][table_name].get(key)

        return dict(self.config[DATASET_TABLES][table_name])

    def get_item_table_info(self, key=None):
        """Gets information of the dataset item table.

        The total dictionary that is available is the same as
        specified in the get_table_info function.

        Args:
            key(str): the specific table information or None for all.

        Returns:
            (Any): the corresponding value to the key when specified or
                a dictionary with all information. When the key is not found or
                there is no item table None is returned.
        """
        return self.get_table_info(
            self.get_item_table_name(),
            key
        )

    def get_item_table_name(self):
        """Gets the name of the item table.

        Returns:
            (str): the item table name.
        """
        return self.config[DATASET_ITEMS]['table']

    def get_user_table_info(self, key=None):
        """Gets information of the dataset user table.

        The total dictionary that is available is the same as
        specified in the get_table_info function.

        Args:
            key(str): the specific table information or None for all.

        Returns:
            (Any): the corresponding value to the key when specified or
                a dictionary with all information. When the key is not found or
                there is no user table None is returned.
        """
        return self.get_table_info(
            self.get_user_table_name(),
            key
        )

    def get_user_table_name(self):
        """Gets the name of the user table.

        Returns:
            (str): the user table name.
        """
        return self.config[DATASET_USERS]['table']

    def load_matrix_df(self):
        """Loads the matrix dataframe.

        Returns:
            (pandas.DataFrame): the loaded matrix.
        """
        names = ['user', 'item', 'rating']
        if self.config[DATASET_MATRIX]['timestamp']:
            names.append('timestamp')

        # TODO synchronize matrix file with lock
        return pd.read_csv(
            self.get_matrix_file_path(),
            sep='\t',
            header=None,
            names=names
        )

    def load_item_indices(self):
        """Loads the item indices.

        Optional indirection array of the item IDs that do not match up in
        the corresponding data table.

        Returns:
            (array like): the indirection array or None when not needed.
        """
        item_idx_file = self.config[DATASET_ITEMS][DATASET_FILE]
        if not item_idx_file:
            return None

        # TODO synchronize item indices file with lock
        return load_array_from_hdf5(
            os.path.join(self.data_dir, item_idx_file),
            DATASET_INDICES
        )

    def load_user_indices(self):
        """Loads the user indices.

        Optional indirection array of the user IDs that do not match up in
        the corresponding data table.

        Returns:
            (array like): the indirection array or None when not needed.
        """
        user_idx_file = self.config[DATASET_USERS][DATASET_FILE]
        if not user_idx_file:
            return None

        # TODO synchronize user indices file with lock
        return load_array_from_hdf5(
            os.path.join(self.data_dir, user_idx_file),
            DATASET_INDICES
        )

    def read_table(self, table_name, columns=None, chunk_size=None):
        """Reads the table with the specified name from the dataset.

        Args:
            table_name(str): name of the table to read.
            columns(array like): subset list of columns to load or None to load all.
                All elements must either be integer indices or
                strings that correspond to the one of the available table columns.
            chunk_size(int): reads the table in chunks as an iterator or
                the entire table when None.

        Returns:
            (pandas.DataFrame): the resulting data table (iterator).
        """
        table_info = self.get_table_info(table_name)
        if not table_info:
            return None

        # TODO synchronize table file(s) with lock(s)
        return read_table(
            self.data_dir,
            table_info,
            columns=columns,
            chunk_size=chunk_size
        )

    def read_item_table(self, columns=None, chunk_size=None):
        """Reads the item table from the dataset.

        Args:
            columns(array like): subset list of columns to load or None to load all.
                All elements must either be integer indices or
                strings that correspond to the one of the available table columns.
            chunk_size(int): reads the table in chunks as an iterator or
                the entire table when None.

        Returns:
            (pandas.DataFrame): the resulting item table (iterator).
        """
        return self.read_table(
            self.get_item_table_name(),
            columns=columns,
            chunk_size=chunk_size
        )

    def read_user_table(self, columns=None, chunk_size=None):
        """Reads the user table from the dataset.

        Args:
            columns(array like): subset list of columns to load or None to load all.
                All elements must either be integer indices or
                strings that correspond to the one of the available table columns.
            chunk_size(int): reads the table in chunks as an iterator or
                the entire table when None.

        Returns:
            (pandas.DataFrame): the resulting user table (iterator).
        """
        return self.read_table(
            self.get_user_table_name(),
            columns=columns,
            chunk_size=chunk_size
        )

    def resolve_item_ids(self, items):
        """Resolves the specified item ID(s).

        The item ID(s) of a dataset need to be resolved when it contains
        an indirection array, otherwise ID(s) are returned unchanged.

        Args:
            items(int or array like): source ID(s) to convert.

        Returns:
            (int or array like): the resolved item ID(s).
        """
        item_indices = self.load_item_indices()
        if item_indices is None:
            return items

        return item_indices[items]

    def resolve_user_ids(self, users):
        """Resolves the specified user ID(s).

        The user ID(s) of a dataset need to be resolved when it contains
        an indirection array, otherwise ID(s) are returned unchanged.

        Args:
            users(int or array like): source ID(s) to convert.

        Returns:
            (int or array like): the resolved user ID(s).
        """
        user_indices = self.load_user_indices()
        if user_indices is None:
            return users

        return user_indices[users]


def add_matrix_columns(dataset, dataframe, column_names):
    """Adds the specified matrix columns to the dataframe.

    Args:
        dataset(Dataset): the set related to the dataframe.
        dataframe(pandas.DataFrame): with at least the 'user' and 'item' columns.
        column_names(array like): list of strings to indicate which
            matrix columns need to be added. Any values that are not
            present in the matrix are ignored.

    Returns:
        (pandas.DataFrame): the resulting dataframe with the added columns.
    """
    # TODO get 'rating' and/or 'timestamp' from the original matrix
    raise NotImplementedError()


def add_user_columns(dataset, dataframe, column_names):
    """Adds the specified user columns to the dataframe.

    Args:
        dataset(Dataset): the set related to the dataframe.
        dataframe(pandas.DataFrame): with at least the 'user' column.
        column_names(array like): list of strings to indicate which
            user columns need to be added. Any values that are not
            present in the user table are ignored.

    Returns:
        (pandas.DataFrame): the resulting dataframe with the added columns or
            when the user table does not exist the unchanged dataframe.
    """
    key = dataset.get_user_table_info('keys')
    if key is None:
        return dataframe

    # resolve user IDs and add them to the dataframe
    key = key[0]
    dataframe[key] = dataset.resolve_user_ids(dataframe['user'])

    # filter any user columns that are not present
    column_names = [c for c in column_names if c in dataset.get_user_table_info('columns')]

    user_table = dataset.read_user_table(columns=[key] + column_names)
    dataframe = pd.merge(dataframe, user_table, how='left', on=key)
    return dataframe.drop(key, axis=1)


def add_item_columns(dataset, dataframe, column_names):
    """Adds the specified item columns to the dataframe.

    Args:
        dataset(Dataset): the set related to the dataframe.
        dataframe(pandas.DataFrame): with at least the 'item' column.
        column_names(array like): list of strings to indicate which
            item columns need to be added. Any values that are not
            present in the item table are ignored.

    Returns:
        (pandas.DataFrame): the resulting dataframe with the added columns or
            when the item table does not exist the unchanged dataframe.
    """
    key = dataset.get_item_table_info('keys')
    if key is None:
        return dataframe

    # resolve item IDs and add them to the dataframe
    key = key[0]
    dataframe[key] = dataset.resolve_item_ids(dataframe['item'])

    # filter any item columns that are not present
    column_names = [c for c in column_names if c in dataset.get_item_table_info('columns')]

    item_table = dataset.read_item_table(columns=[key] + column_names)
    dataframe = pd.merge(dataframe, item_table, how='left', on=key)
    return dataframe.drop(key, axis=1)
