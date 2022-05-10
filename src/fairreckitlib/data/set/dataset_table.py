"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from ..utility import load_table
from .dataset_config import DATASET_FILE


def create_table_config(file_name, keys, columns, header=False, sep=None, encoding='utf-8'):
    """Creates a data table configuration.

    Wraps a table configuration to a standard format. The 'keys' are expected to be in order and
    at the beginning of each row, while the 'columns' follow thereafter in order as well.
    Effectively each row is the concatenation of 'keys' with 'columns'.

    Args:
        file_name(str): name of the data table file.
        keys(array like): list of strings that are used as keys in the table.
        columns(array like): list of strings with other available columns in the table.
        header(bool): whether the table file contains a header on the first line.
        sep(str): the delimiter that is used in the table or None for a tab separator.
        encoding(str): the encoding to use for reading/writing the table contents.

    Returns:
        (dict): the resulting data table configuration.
    """
    return {
        DATASET_FILE: file_name,
        'columns': columns,
        'encoding': encoding,
        'header': header,
        'keys': keys,
        'sep': sep
    }


def read_table(table_dir, table_config, columns=None, chunk_size=None):
    """Reads a data table from a file.

    Args:
        table_dir(str): directory path to where the table file is stored.
        table_config(dict): dictionary containing the table configuration.
        columns(array like): subset list of columns to load or None to load all.
            All elements must either be integer indices or
            strings that correspond to the 'names' argument.
        chunk_size(int): loads the table in chunks as an iterator or
            the entire table when None.

    Returns:
        (pandas.DataFrame): the resulting data table (iterator).
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


def write_table(table, table_dir, table_name, sep=None, encoding='utf-8'):
    """Writes a data table to a file.

    Args:
        table(pandas.DataFrame): the source table to save.
        table_dir(str): directory path to where the table file will be stored.
        table_name(str): name of the data table file that will be stored.
        sep(str): the delimiter that is used in the table or None for a tab separator.
        encoding(str): the encoding to use for writing the table contents.

    Returns:
        (pandas.DataFrame): the resulting data table.
    """
    table.to_csv(
        os.path.join(table_dir, table_name),
        sep=sep if sep else '\t',
        header=False,
        index=False,
        encoding=encoding
    )
