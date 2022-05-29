"""This module contains the dataset configuration classes and creation functions.

Constants:

    DATASET_RATINGS_EXPLICIT: dataset matrix with explicit ratings.
    DATASET_RATINGS_IMPLICIT: dataset matrix with implicit ratings.

Classes:

    DatasetFileConfig: the configuration of a dataset file.
    DatasetTableConfig: the configuration of a dataset table.
    DatasetIndexConfig: the configuration of a dataset matrix' user/item indices.
    DatasetMatrixConfig: the configuration of a dataset matrix.
    DatasetConfig: the configuration of a dataset.

Functions:

    create_dataset_table_config: create configuration for a dataset table.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ...core.config_constants import KEY_NAME
from ...core.config_object import YmlConfig, format_yml_config_dict
from ..utility import load_array_from_hdf5, save_array_to_hdf5
from .dataset_constants import KEY_RATING_MIN, KEY_RATING_MAX, KEY_RATING_TYPE
from .dataset_constants import KEY_MATRIX, KEY_IDX_ITEM, KEY_IDX_USER
from .dataset_constants import KEY_DATASET, KEY_EVENTS, KEY_MATRICES, KEY_TABLES
from .dataset_constants import TABLE_KEY, TABLE_PRIMARY_KEY, TABLE_FOREIGN_KEYS, TABLE_COLUMNS
from .dataset_constants import TABLE_FILE, TABLE_COMPRESSION, TABLE_ENCODING
from .dataset_constants import TABLE_HEADER, TABLE_INDEXED, TABLE_NUM_RECORDS, TABLE_SEP

DATASET_RATINGS_EXPLICIT = 'explicit'
DATASET_RATINGS_IMPLICIT = 'implicit'


@dataclass
class DatasetFileConfig(YmlConfig):
    r"""Dataset File Configuration.

    name: the file name.
    sep: the separator in the file or None for \t.
    compression: the (optional) compression of the file.
    encoding: the encoding of the file or None for 'utf-8'.
    header: is there a header on the first line of the file.
    indexed: are the row indices the table's primary key.
    """

    name: str
    sep: Optional[str]
    compression: Optional[str]
    encoding: Optional[str]
    header: bool
    indexed: bool

    def to_yml_format(self):
        """Format dataset file configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset file configuration.
        """
        yml_format = {KEY_NAME: self.name}

        if self.header:
            yml_format[TABLE_HEADER] = self.header
        if self.indexed:
            yml_format[TABLE_INDEXED] = self.indexed
        if self.sep is not None:
            yml_format[TABLE_SEP] = self.sep
        if self.compression is not None:
            yml_format[TABLE_COMPRESSION] = self.compression
        if self.encoding is not None:
            yml_format[TABLE_ENCODING] = self.encoding

        return yml_format


@dataclass
class DatasetTableConfig(YmlConfig):
    """Dataset Table Configuration.

    The configuration expects the table to have the primary key as the first column(s) and in order
    in which they are specified. These are followed by the columns of relevant data and any foreign
    keys should be in order as the last column(s). However, it is also allowed that the foreign
    keys describe the primary key, but individually rather than the combination of.

    primary_key: list of column names that form the primary key of the table.
    foreign_keys: (optional) list of column names that are foreign keys in other tables.
    columns: list of column names that contain the relevant table data.
    num_records: the number of records in the table.
    file: the dataset file configuration of the table.
    """

    primary_key: List[str]
    foreign_keys: Optional[List[str]]
    columns: List[str]
    num_records: int
    file: DatasetFileConfig

    def read_table(
            self,
            dataset_dir: str,
            *,
            columns: List[Union[str, int]]=None,
            chunk_size=None) -> pd.DataFrame:
        """Read the table from the specified directory.

        Args:
            dataset_dir: the directory to read the table from.
            columns: subset list of columns to load or None to load all.
                All elements must either be integer indices or
                strings that correspond to the 'names' argument.
            chunk_size: loads the table in chunks as an iterator or
                the entire table when None.

        Returns:
            the resulting table (iterator).
        """
        names = self.columns
        if not self.file.indexed:
            names = self.primary_key + names
        if self.foreign_keys is not None:
            names += [key for key in self.foreign_keys if key not in self.primary_key]

        dataset_table = pd.read_table(
            os.path.join(dataset_dir, self.file.name),
            sep=self.file.sep if self.file.sep is not None else '\t',
            header=0 if self.file.header else None,
            names=names,
            usecols=columns,
            encoding=self.file.encoding if self.file.encoding is not None else 'utf-8',
            compression=self.file.compression if self.file.compression is not None else 'infer',
            chunksize=chunk_size,
            iterator=bool(chunk_size)
        )

        return dataset_table

    def save_table(self, dataset_table: pd.DataFrame, dataset_dir: str) -> None:
        """Save the table in the specified directory.

        Args:
            dataset_table: the dataframe to save with this table configuration.
            dataset_dir: the directory to save the table to.
        """
        dataset_table.to_csv(
            os.path.join(dataset_dir, self.file.name),
            sep=self.file.sep if self.file.sep else '\t',
            header=self.file.header,
            index=False,
            encoding=self.file.encoding if self.file.encoding else 'utf-8',
            compression=self.file.compression if self.file.compression else 'infer',
        )

    def to_yml_format(self) -> Dict[str, Any]:
        """Format dataset table configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset table configuration.
        """
        yml_format = {
            TABLE_FILE: self.file.to_yml_format(),
            TABLE_PRIMARY_KEY: self.primary_key,
            TABLE_COLUMNS: self.columns,
            TABLE_NUM_RECORDS: self.num_records,
        }

        if self.foreign_keys is not None:
            yml_format[TABLE_FOREIGN_KEYS] = self.foreign_keys

        return yml_format


@dataclass
class DatasetIndexConfig(YmlConfig):
    """Dataset Matrix' Index Configuration.

    file_name: (optional) file name that contains the user/item indirection array.
    key: the key that is associated with the user/item.
    num_records: the number of user/item records
    """

    file_name: Optional[str]
    key: str
    num_records: int

    def load_indices(self, dataset_dir: str) -> Optional[List[int]]:
        """Load the indices from the specified directory.

        This function raises a FileNotFoundError when the file is not
        found in the specified directory.

        Args:
            dataset_dir: the directory to load the indices from.

        Returns:
            the resulting indices or None when not available.
        """
        if self.file_name is None:
            return None

        return load_array_from_hdf5(os.path.join(dataset_dir, self.file_name), 'indices')

    def save_indices(self, dataset_dir: str, indices: List[int]) -> bool:
        """Save the indices to the specified directory.

        Args:
            dataset_dir: the directory to save the indices to.
            indices: the list of indices to save.

        Returns:
            true when the indices are saved or false when the configuration has no file name.
        """
        if self.file_name is None:
            return False

        save_array_to_hdf5(os.path.join(dataset_dir, self.file_name), indices, 'indices')
        return True

    def to_yml_format(self) -> Dict[str, Any]:
        """Format dataset index configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset index configuration.
        """
        yml_format = {
            TABLE_KEY: self.key,
            TABLE_NUM_RECORDS: self.num_records
        }

        if self.file_name is not None:
            yml_format[TABLE_FILE] = self.file_name

        return yml_format


@dataclass
class DatasetMatrixConfig(YmlConfig):
    """Dataset Matrix Configuration.

    table: the table configuration of the matrix.
    rating_min: the minimum rating in the matrix.
    rating_max: the maximum rating in the matrix.
    rating_type: the type of the rating in the matrix, either 'explicit' or 'implicit'.
    user: the dataset index configuration for the users in the matrix.
    item: the dataset index configuration for the items in the matrix.
    """

    table: DatasetTableConfig
    rating_min: float
    rating_max: float
    rating_type: str
    user: DatasetIndexConfig
    item: DatasetIndexConfig

    def load_matrix(self, dataset_dir: str) -> pd.DataFrame:
        """Load the matrix from the specified directory.

        Args:
            dataset_dir: directory path to where the dataset matrix is stored.

        Returns:
            the resulting matrix (iterator).
        """
        matrix = self.table.read_table(dataset_dir)

        columns = {
            self.user.key: 'user',
            self.item.key: 'item',
            self.table.columns[0]: 'rating'
        }
        if len(self.table.columns) == 2:
            columns[self.table.columns[1]] = 'timestamp'

        return matrix.rename(columns=columns)

    def to_yml_format(self) -> Dict[str, Any]:
        """Format dataset matrix configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset matrix configuration.
        """
        return {
            KEY_IDX_ITEM: self.item.to_yml_format(),
            KEY_IDX_USER: self.user.to_yml_format(),
            KEY_MATRIX: self.table.to_yml_format(),
            KEY_RATING_MIN: self.rating_min,
            KEY_RATING_MAX: self.rating_max,
            KEY_RATING_TYPE: self.rating_type,
        }


@dataclass
class DatasetConfig(YmlConfig):
    """Dataset Configuration.

    dataset_name: the name of the dataset.
    events: dictionary containing the available user event tables.
    matrices: dictionary containing the available user-item matrices.
    tables: dictionary containing the (additionally) available tables.
    """

    dataset_name: str
    events: Dict[str, DatasetTableConfig]
    matrices: Dict[str, DatasetMatrixConfig]
    tables: Dict[str, DatasetTableConfig]

    def get_available_columns(self, matrix_name: str) -> Dict[str, List[str]]:
        """Get the available columns of the specified matrix.

        Only the table names and columns that have a one-to-one relation will be returned.
        This function does not raise errors and will return an empty dictionary when
        the specified matrix is not present in the dataset.

        Args:
            matrix_name: the name of the matrix to get the available columns of.

        Returns:
            a dictionary containing the table names as key and the available columns as value.
        """
        if matrix_name not in self.matrices:
            return {}

        matrix_config = self.matrices[matrix_name]
        result = {KEY_MATRIX: matrix_config.table.columns}

        user_key = [matrix_config.user.key]
        item_key = [matrix_config.item.key]
        user_item_key = [user_key, item_key]

        for table_name, table_config in self.tables.items():
            key = table_config.primary_key
            if key in (user_key, item_key, user_item_key):
                result[table_name] = table_config.columns

        return result

    def to_yml_format(self) -> Dict[str, Any]:
        """Format dataset configuration to a yml compatible dictionary.

        Returns:
            a dictionary containing the dataset configuration.
        """
        yml_format = {KEY_DATASET: self.dataset_name}

        if len(self.events) > 0:
            yml_format[KEY_EVENTS] = format_yml_config_dict(self.events)
        if len(self.matrices) > 0:
            yml_format[KEY_MATRICES] = format_yml_config_dict(self.matrices)
        if len(self.tables) > 0:
            yml_format[KEY_TABLES] = format_yml_config_dict(self.tables)

        return yml_format


def create_dataset_table_config(
        file_name: str,
        primary_key: List[str],
        columns: List[str],
        *,
        compression: str=None,
        encoding: str=None,
        foreign_keys: List[str]=None,
        header: bool=False,
        indexed: bool=False,
        num_records: int=0,
        sep: str=None) -> DatasetTableConfig:
    """Create a dataset table configuration.

    Args:
        file_name: name of the dataset table file.
        primary_key: a list of strings that are combined the primary key of the table.
        columns: a list of strings with other available columns in the table.
        compression:  the (optional) compression of the file, 'bz2' is recommended.
        encoding: the encoding for reading/writing the table contents or None for 'utf-8'.
        foreign_keys: (optional) list of column names that are foreign keys in other tables.
        header: whether the table file contains a header on the first line.
        indexed: are the row indices the table's primary key.
        num_records: the number of records in the table.
        sep: the delimiter that is used in the table or None for a tab separator.

    Returns:
        the resulting data table configuration.
    """
    return DatasetTableConfig(
        primary_key,
        foreign_keys,
        columns,
        num_records,
        DatasetFileConfig(
            file_name,
            sep,
            compression,
            encoding,
            header,
            indexed
        )
    )
