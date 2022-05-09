"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import h5py
import numpy as np
import pandas as pd
import yaml


def convert_csr_to_coo(csr_matrix):
    """Converts a scipy CSR sparse matrix to COO format.

    Arg:
        csr_matrix(scipy.sparse.csr_matrix): source matrix to convert.

    Returns:
        matrix(pandas.sparse.DataFrame) the resulting COO matrix.
    """
    matrix = pd.DataFrame.sparse.from_spmatrix(csr_matrix)
    return matrix.sparse.to_coo()


def convert_coo_to_df(coo_matrix, row_name, col_name, val_name):
    """Converts a pandas COO matrix to a regular DataFrame.

    Args:
        coo_matrix(pandas.sparse.DataFrame) source COO matrix to convert.
        row_name(str): name of the COO row in the resulting dataframe.
        col_name(str): name of the COO column in the resulting dataframe.
        val_name(str): name of the COO value in the resulting dataframe.

    Returns:
        dataframe(pandas.DataFrame): the resulting dataframe.
    """
    dataframe = pd.DataFrame()
    dataframe[row_name] = coo_matrix.row
    dataframe[col_name] = coo_matrix.col
    dataframe[val_name] = coo_matrix.data
    return dataframe


def load_array_from_hdf5(file_path, array_name):
    """Loads a single array from a HDF5 binary data file.

    Counterpart of the save_array_to_hdf5 function.

    Args:
        file_path(str): path to where the HDF5 file is stored.
        array_name(str): name of the array to retrieve from the file.

    Returns:
        (numpy.array): the array data from the file.
    """
    with h5py.File(file_path, "r") as file:
        return np.array(file.get(array_name))


def load_df_from_hdf5(file_path):
    """Loads a dataframe from a HDF5 binary data file.

    Counterpart of the save_df_to_hdf5 function.

    Args:
        file_path(str): path to where the HDF5 file is stored.

    Returns:
        (pandas.DataFrame): the dataframe from the file.
    """
    with h5py.File(file_path, "r") as file:
        dataframe = pd.DataFrame()

        for key, values in file.items():
            dataframe[key] = values

        return dataframe


def load_table(file_path, names, columns=None, header=False, sep=None,
               encoding='utf-8', chunk_size=None):
    """Loads a table from a file.

    Args:
        file_path(str): path to where the table file is stored.
        names(array like): list of all column names in the table.
        columns(array like): subset list of columns to load or None to load all.
            All elements must either be integer indices or
            strings that correspond to the 'names' argument.
        header(bool): whether the table file contains a header on the first line.
        sep(str): the delimiter that is used in the table or None for a tab separator.
        encoding(str): the encoding to use for reading the table contents.
        chunk_size(int): loads the table in chunks as an iterator or
            the entire table when None.

    Returns:
        (pandas.DataFrame): the resulting table.
    """
    return pd.read_table(
        file_path,
        header=0 if header else None,
        sep=sep if sep else '\t',
        names=names,
        usecols=columns,
        encoding=encoding,
        chunksize=chunk_size,
        iterator=bool(chunk_size)
    )


def load_yml(file_path, encoding='utf-8'):
    """Loads a yml file.

    Counterpart of the save_yml function.

    Args:
        file_path(str): path to where the yml file is stored.
        encoding(str): the encoding to use for reading the file contents.

    Returns:
        (dict): the resulting dictionary.
    """
    with open(file_path, 'r', encoding=encoding) as yml_file:
        return yaml.safe_load(yml_file)


def save_array_to_hdf5(file_path, arr, array_name):
    """Saves a single array to a HDF5 binary data file.

    Counterpart of the load_array_from_hdf5 function.

    Args:
        file_path(str): path to where the HDF5 file will be stored.
        arr(array like): the source array to save in the file.
        array_name(str): name of the array to save in the file.
    """
    with h5py.File(file_path, "w") as file:
        file.create_dataset(array_name, data=arr)


def save_df_to_hdf5(file_path, dataframe):
    """Saves a dataframe to a HDF5 binary data file.

    Counterpart of the load_array_from_hdf5 function.

    Args:
        file_path(str): path to where the HDF5 file will be stored.
        dataframe(pandas.DataFrame): the source dataframe to save in the file.
    """
    with h5py.File(file_path, "w") as file:
        for key in dataframe:
            file.create_dataset(key, data=dataframe[key])


def save_yml(file_path, config, encoding='utf-8'):
    """Saves a yml file.

    Counterpart of the load_yml function.

    Args:
        file_path(str): path to where the yml file will be stored.
        config(dict): the source dictionary to save in the file.
        encoding(str): the encoding to use for reading the file contents.
    """
    with open(file_path, 'w', encoding=encoding) as yml_file:
        yaml.dump(config, yml_file)
