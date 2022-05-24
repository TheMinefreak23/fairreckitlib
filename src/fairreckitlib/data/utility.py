"""This module contains data utility functions.

Functions:

    convert_csr_to_coo: convert csr matrix to coo matrix.
    convert_coo_to_df: convert coo matrix to dataframe.
    load_array_from_hdf5: load array data from hdf5 file.
    load_df_from_hdf5: load dataframe from hdf5 file.
    load_table: load data table from file.
    load_yml: load yml file.
    save_array_to_hdf5: save array data to hdf5 file.
    save_df_to_hdf5: save dataframe to hdf5 file.
    save_yml: save yml file.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import yaml


def convert_csr_to_coo(csr_matrix: sparse.csr_matrix) -> sparse.coo_matrix:
    """Convert a scipy CSR sparse matrix to COO format.

    Arg:
        csr_matrix: source matrix to convert.

    Returns:
        the resulting COO matrix.
    """
    matrix = pd.DataFrame.sparse.from_spmatrix(csr_matrix)
    return matrix.sparse.to_coo()


def convert_coo_to_df(
        matrix: sparse.coo_matrix,
        row_name: str,
        col_name: str,
        val_name: str) -> pd.DataFrame:
    """Convert a pandas COO matrix to a regular DataFrame.

    Args:
        matrix: source COO matrix to convert.
        row_name: name of the COO row in the resulting dataframe.
        col_name: name of the COO column in the resulting dataframe.
        val_name: name of the COO value in the resulting dataframe.

    Returns:
        the resulting dataframe.
    """
    dataframe = pd.DataFrame()
    dataframe[row_name] = matrix.row
    dataframe[col_name] = matrix.col
    dataframe[val_name] = matrix.data
    return dataframe


def load_array_from_hdf5(file_path: str, array_name: str) -> np.array:
    """Load a single array from a HDF5 binary data file.

    Counterpart of the save_array_to_hdf5 function.

    Args:
        file_path: path to where the HDF5 file is stored.
        array_name: name of the array to retrieve from the file.

    Returns:
        the array data from the file.
    """
    with h5py.File(file_path, "r") as file:
        return np.array(file.get(array_name))


def load_df_from_hdf5(file_path: str) -> pd.DataFrame:
    """Load a dataframe from a HDF5 binary data file.

    Counterpart of the save_df_to_hdf5 function.

    Args:
        file_path: path to where the HDF5 file is stored.

    Returns:
        the dataframe from the file.
    """
    with h5py.File(file_path, "r") as file:
        dataframe = pd.DataFrame()

        for key, values in file.items():
            dataframe[key] = values

        return dataframe


def load_yml(file_path: str, encoding: str='utf-8') -> Dict:
    """Load a yml file.

    Counterpart of the save_yml function.

    Args:
        file_path: path to where the yml file is stored.
        encoding: the encoding to use for reading the file contents.

    Returns:
        the resulting dictionary.
    """
    with open(file_path, 'r', encoding=encoding) as yml_file:
        return yaml.safe_load(yml_file)


def save_array_to_hdf5(file_path: str, arr: List[Any], array_name: str) -> None:
    """Save a single array to a HDF5 binary data file.

    Counterpart of the load_array_from_hdf5 function.

    Args:
        file_path: path to where the HDF5 file will be stored.
        arr: the source array to save in the file.
        array_name: name of the array to save in the file.
    """
    with h5py.File(file_path, "w") as file:
        file.create_dataset(array_name, data=arr)


def save_df_to_hdf5(file_path: str, dataframe: pd.DataFrame) -> None:
    """Save a dataframe to a HDF5 binary data file.

    Counterpart of the load_array_from_hdf5 function.

    Args:
        file_path: path to where the HDF5 file will be stored.
        dataframe: the source dataframe to save in the file.
    """
    with h5py.File(file_path, "w") as file:
        for key in dataframe:
            file.create_dataset(key, data=dataframe[key])


def save_yml(file_path: str, config: Dict, encoding: str='utf-8') -> None:
    """Save a yml file.

    Counterpart of the load_yml function.

    Args:
        file_path: path to where the yml file will be stored.
        config: the source dictionary to save in the file.
        encoding: the encoding to use for reading the file contents.
    """
    with open(file_path, 'w', encoding=encoding) as yml_file:
        yaml.dump(config, yml_file)
