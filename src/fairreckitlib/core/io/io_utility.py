"""This module contains IO utility functions that connect with external packages.

Functions:

    load_array_from_hdf5: load array data from hdf5 file.
    load_json: load dictionary from json file.
    load_yml: load dictionary from yml file.
    save_array_to_hdf5: save array data to hdf5 file.
    save_json: save dictionary to json file.
    save_yml: save dictionary to yml file.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import json
from typing import Any, Dict, List, Union

import h5py
import numpy as np
import yaml


def load_array_from_hdf5(file_path: str, array_name: str) -> np.array:
    """Load a single array from a HDF5 binary data file.

    This function raises a FileNotFoundError when the specified file does not exist.
    Counterpart of the save_array_to_hdf5 function.

    Args:
        file_path: path to where the HDF5 file is stored.
        array_name: name of the array to retrieve from the file.

    Returns:
        the array data from the file.
    """
    with h5py.File(file_path, "r") as file:
        return np.array(file.get(array_name))


def load_json(file_path: str, encoding: str='utf-8') -> Union[Dict[str, Any], List]:
    """Load a json file.

    This function raises a FileNotFoundError when the specified file does not exist.
    Counterpart of the save_json function.

    Args:
        file_path: path to where the json file is stored.
        encoding: the encoding to use for reading the file contents.

    Returns:
        the resulting dictionary.
    """
    with open(file_path, encoding=encoding) as out_file:
        return json.load(out_file)


def load_yml(file_path: str, encoding: str='utf-8') -> Union[Dict[str, Any], List]:
    """Load a yml file.

    This function raises a FileNotFoundError when the specified file does not exist.
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


def save_json(file_path: str, data: Union[Dict[str, Any], List],
              *, encoding: str='utf-8', indent=None) -> None:
    """Save a json file.

    Counterpart of the load_json function.

    Args:
        file_path: path to where the json file will be stored.
        data: the source dictionary to save in the file.
        encoding: the encoding to use for writing the file contents.
        indent: the indent level for pretty printing JSON array elements and object members.
    """
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(data, file, indent=indent)


def save_yml(file_path: str, data: Union[Dict[str, Any], List], *, encoding: str='utf-8') -> None:
    """Save a yml file.

    Counterpart of the load_yml function.

    Args:
        file_path: path to where the yml file will be stored.
        data: the source dictionary to save in the file.
        encoding: the encoding to use for writing the file contents.
    """
    with open(file_path, 'w', encoding=encoding) as yml_file:
        yaml.dump(data, yml_file)
