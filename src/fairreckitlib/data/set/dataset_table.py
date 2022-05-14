"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict, List, Union

import pandas as pd

from ..utility import load_table
from .dataset_config import DATASET_FILE


def create_table_config(
        file_name: str,
        keys: List[str],
        columns: List[str],
        header: bool=False,
        sep: str=None,
        encoding: str='utf-8') -> Dict[str, Any]:
    """Create a data table configuration.

    Wraps a table configuration to a standard format. The 'keys' are expected to be in order and
    at the beginning of each row, while the 'columns' follow thereafter in order as well.
    Effectively each row is the concatenation of 'keys' with 'columns'.

    Args:
        file_name: name of the data table file.
        keys: a list of strings that are used as keys in the table.
        columns: a list of strings with other available columns in the table.
        header: whether the table file contains a header on the first line.
        sep: the delimiter that is used in the table or None for a tab separator.
        encoding: the encoding to use for reading/writing the table contents.

    Returns:
        the resulting data table configuration.
    """
    return {
        DATASET_FILE: file_name,
        'columns': columns,
        'encoding': encoding,
        'header': header,
        'keys': keys,
        'sep': sep
    }


def read_table(
        table_dir: str,
        table_config: Dict[str, Any],
        columns: List[Union[int,str]]=None,
        chunk_size: int=None) -> pd.DataFrame:
    """Read a data table from a file.

    Args:
        table_dir: directory path to where the table file is stored.
        table_config: dictionary containing the table configuration.
        columns: subset list of columns to load or None to load all.
            All elements must either be integer indices or
            strings that correspond to the 'names' argument.
        chunk_size: loads the table in chunks as an iterator or
            the entire table when None.

    Returns:
        the resulting data table (iterator).
    """
    return load_table(
        os.path.join(table_dir, table_config[DATASET_FILE]),
        table_config['keys'] + table_config['columns'],
        columns=columns,
        header=table_config['header'],
        sep=table_config['sep'],
        encoding=table_config['encoding'],
        chunk_size=chunk_size
    )


def write_table(
        table: pd.DataFrame,
        table_dir: str,
        table_name: str,
        sep: str=None,
        encoding: str='utf-8') -> None:
    """Write a data table to a file.

    Args:
        table: the source table to save.
        table_dir: directory path to where the table file will be stored.
        table_name: name of the data table file that will be stored.
        sep: the delimiter that is used in the table or None for a tab separator.
        encoding: the encoding to use for writing the table contents.
    """
    table.to_csv(
        os.path.join(table_dir, table_name),
        sep=sep if sep else '\t',
        header=False,
        index=False,
        encoding=encoding
    )
