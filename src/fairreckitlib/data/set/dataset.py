"""This module contains a dataset definition for accessing a dataset and related data tables.

Classes:

    Dataset: class wrapper of the user events, user-item matrices and related tables.

Functions:

    add_dataset_columns: add columns from the dataset matrix/user/item tables to a dataframe.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict, Optional, List, Union

import pandas as pd

from .dataset_config import DatasetConfig, DatasetMatrixConfig, DatasetTableConfig


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

    Public methods:

    get_available_columns
    get_available_matrices
    get_available_tables
    get_matrices_info
    get_matrix_config
    get_matrix_file_path
    get_name
    get_table_config
    get_table_info
    load_matrix
    read_matrix
    read_table
    resolve_item_ids
    resolve_user_ids
    """

    def __init__(self, data_dir: str, config: DatasetConfig):
        """Construct the dataset.

        Args:
            data_dir: directory where the dataset is stored.
            config: data configuration dictionary.
        """
        self.data_dir = data_dir
        self.config = config

    def get_available_columns(self, matrix_name: str) -> Dict[str, List[str]]:
        """Get the available table column names of this dataset.

        Args:
            matrix_name: the name of the matrix to get the available columns of.

        Returns:
            a dictionary with table name as keys and column names as values.

        """
        return self.config.get_available_columns(matrix_name)

    def get_available_matrices(self) -> List[str]:
        """Get the available matrix names in the dataset.

        Returns:
            a list of matrix names.
        """
        matrix_names = []

        for matrix_name, _ in self.config.matrices.items():
            matrix_names.append(matrix_name)

        return matrix_names

    def get_available_tables(self) -> List[str]:
        """Get the available table names in the dataset.

        Returns:
            a list of table names.
        """
        table_names = []

        for table_name, _ in self.config.tables.items():
            table_names.append(table_name)

        return table_names

    def get_matrices_info(self) -> Dict[str, Any]:
        """Get the information on the dataset's available matrices.

        Returns:
            a dictionary containing the matrices' information keyed by matrix name.
        """
        info = {}

        for matrix_name, matrix_config in self.config.matrices.items():
            info[matrix_name] = matrix_config.to_yml_format()

        return info

    def get_matrix_config(self, matrix_name: str) -> Optional[DatasetMatrixConfig]:
        """Get the configuration of a dataset's matrix.

        Args:
            matrix_name: the name of the matrix to get the configuration of.

        Returns:
            the configuration of the matrix or None when not available.
        """
        return self.config.matrices.get(matrix_name)

    def get_matrix_file_path(self, matrix_name: str) -> Optional[str]:
        """Get the file path where the matrix with the specified name is stored.

        Args:
            matrix_name: the name of the matrix to get the file path of.

        Returns:
            the path of the dataset's matrix file or None when not available.
        """
        if matrix_name not in self.config.matrices:
            return None

        return os.path.join(
            self.data_dir,
            self.config.matrices[matrix_name].table.file.name
        )

    def get_name(self) -> str:
        """Get the name of the dataset.

        Returns:
            the dataset name.
        """
        return self.config.dataset_name

    def get_table_config(self, table_name: str) -> Optional[DatasetTableConfig]:
        """Get the configuration of the dataset table with the specified name.

        Args:
            table_name: name of the table to retrieve the configuration of.

        Returns:
            the table configuration or None when not available.
        """
        return self.config.tables.get(table_name)

    def get_table_info(self) -> Dict[str, Any]:
        """Get the information on the dataset's available tables.

        Returns:
            a dictionary containing the table information keyed by table name.
        """
        info = {}

        for table_name, table_config in self.config.tables.items():
            info[table_name] = table_config.to_yml_format()

        return info

    def load_matrix(self, matrix_name: str) -> Optional[pd.DataFrame]:
        """Load the standardized user-item matrix of the dataset.

        Args:
            matrix_name: the name of the matrix to load.

        Returns:
            the loaded user-item matrix or None when not available.
        """
        matrix_config = self.get_matrix_config(matrix_name)
        if matrix_config is None:
            return None

        return matrix_config.load_matrix(self.data_dir)

    def load_item_indices(self, matrix_name: str) -> Optional[List[int]]:
        """Load the item indices.

        Optional indirection array of the item IDs that do not match up in
        the corresponding data table.

        Args:
            matrix_name: the name of the matrix to load the item indices of.

        Returns:
            the indirection array or None when not needed.
        """
        matrix = self.config.matrices.get(matrix_name)
        if not matrix:
            return None

        return matrix.item.load_indices(self.data_dir)

    def load_user_indices(self, matrix_name: str) -> Optional[List[int]]:
        """Load the user indices.

        Optional indirection array of the user IDs that do not match up in
        the corresponding data table.

        Args:
            matrix_name: the name of the matrix to load the user indices of.

        Returns:
            the indirection array or None when not needed.
        """
        matrix = self.config.matrices.get(matrix_name)
        if not matrix:
            return None

        return matrix.user.load_indices(self.data_dir)

    def read_matrix(
            self,
            matrix_name: str,
            columns: List[Union[int,str]]=None,
            chunk_size: int=None) -> Optional[pd.DataFrame]:
        """Read the matrix with the specified name from the dataset.

        Args:
            matrix_name: the name of the matrix to load.
            columns: subset list of columns to load or None to load all.
                All elements must either be integer indices or
                strings that correspond to the one of the available table columns.
            chunk_size: reads the matrix in chunks as an iterator or
                the entire table when None.

        Returns:
            the resulting matrix dataframe (iterator) or None when not available.
        """
        matrix_config = self.get_matrix_config(matrix_name)
        if matrix_config is None:
            return None

        table_config = matrix_config.table
        return table_config.read_table(self.data_dir, columns=columns, chunk_size=chunk_size)

    def read_table(
            self,
            table_name: str,
            columns: List[Union[int,str]]=None,
            chunk_size: int=None) -> Optional[pd.DataFrame]:
        """Read the table with the specified name from the dataset.

        Args:
            table_name: name of the table to read.
            columns: subset list of columns to load or None to load all.
                All elements must either be integer indices or
                strings that correspond to the one of the available table columns.
            chunk_size: reads the table in chunks as an iterator or
                the entire table when None.

        Returns:
            the resulting table dataframe (iterator) or None when not available.
        """
        table_config = self.get_table_config(table_name)
        if table_config is None:
            return None

        return table_config.read_table(self.data_dir, columns=columns, chunk_size=chunk_size)

    def resolve_item_ids(
            self,
            matrix_name: str,
            items: Union[int,List[int]]) -> Union[int,List[int]]:
        """Resolve the specified item ID(s).

        The item ID(s) of a dataset need to be resolved when it contains
        an indirection array, otherwise ID(s) are returned unchanged.

        Args:
            matrix_name: the name of the matrix to resolve the item indices of.
            items: source ID(s) to convert.

        Returns:
            the resolved item ID(s).
        """
        item_indices = self.load_item_indices(matrix_name)
        if item_indices is None:
            return items

        return item_indices[items]

    def resolve_user_ids(
            self,
            matrix_name: str,
            users: Union[int,List[int]]) -> Union[int,List[int]]:
        """Resolve the specified user ID(s).

        The user ID(s) of a dataset need to be resolved when it contains
        an indirection array, otherwise ID(s) are returned unchanged.

        Args:
            matrix_name: the name of the matrix to resolve the user indices of.
            users: source ID(s) to convert.

        Returns:
            the resolved user ID(s).
        """
        user_indices = self.load_user_indices(matrix_name)
        if user_indices is None:
            return users

        return user_indices[users]


def add_dataset_columns(
        dataset: Dataset,
        matrix_name: str,
        dataframe: pd.DataFrame,
        column_names: List[str]) -> pd.DataFrame:
    """Add the specified columns from the dataset to the dataframe.

    Args:
        dataset: the set related to the dataframe.
        matrix_name: the name of the dataset matrix.
        dataframe: with at least the 'user' and/or 'item' columns.
        column_names: a list of strings to indicate which
            user and/or item columns need to be added. Any values that are not
            present in the dataset tables are ignored.

    Returns:
        the resulting dataframe with the added columns that exist in the dataset.
    """
    for table_name, table_columns in dataset.get_available_columns(matrix_name).items():
        columns = [c for c in column_names if c in table_columns]
        # skip table that does not contain any needed columns
        if len(columns) == 0:
            continue

        matrix_config = dataset.get_matrix_config(matrix_name)
        table_config = dataset.get_table_config(table_name)

        user_key = matrix_config.user.key
        item_key = matrix_config.item.key
        user_item_key = [user_key, item_key]

        # add matrix columns
        if table_name == 'matrix':
            dataframe = pd.merge(
                dataframe,
                dataset.read_matrix(matrix_name, columns=matrix_config.table.primary_key + columns),
                how='left',
                left_on=['user', 'item'],
                right_on=matrix_config.table.primary_key
            )
            dataframe.drop(matrix_config.table.primary_key, inplace=True, axis=1)
        # add user columns
        elif table_config.primary_key == [user_key]:
            dataframe[user_key] = dataset.resolve_user_ids(matrix_name, dataframe['user'])
            dataframe = pd.merge(
                dataframe,
                dataset.read_table(table_name, columns=table_config.primary_key + columns),
                how='left',
                on=user_key
            )
            dataframe.drop(user_key, inplace=True, axis=1)
        # add item columns
        elif table_config.primary_key == [item_key]:
            dataframe[item_key] = dataset.resolve_item_ids(matrix_name, dataframe['item'])
            dataframe = pd.merge(
                dataframe,
                dataset.read_table(table_name, columns=table_config.primary_key + columns),
                how='left',
                on=item_key
            )
            dataframe.drop(item_key, inplace=True, axis=1)
        # add user-item columns
        elif table_config.primary_key == user_item_key:
            dataframe[user_key] = dataset.resolve_user_ids(matrix_name, dataframe['user'])
            dataframe[item_key] = dataset.resolve_item_ids(matrix_name, dataframe['item'])
            dataframe = pd.merge(
                dataframe,
                dataset.read_table(table_name, columns=table_config.primary_key + columns),
                how='left',
                on=user_item_key
            )
            dataframe.drop(user_item_key, inplace=True, axis=1)

    return dataframe
