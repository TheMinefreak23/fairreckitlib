"""This module tests the core IO functionality.

Functions:

    test_io_create_and_delete_dir: test creation and deletion of (nested) directories.
    test_io_json_and_yml: test the save/load functions from yml and json.
    test_io_hdf5_array: test the save/load hdf5 array functions.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Any, Dict, Callable, List, Union

import numpy as np
import pytest

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.io.io_create import create_dir, create_json, create_yml
from src.fairreckitlib.core.io.io_delete import delete_dir, delete_file
from src.fairreckitlib.core.io.io_utility import load_json, load_yml
from src.fairreckitlib.core.io.io_utility import load_array_from_hdf5, save_array_to_hdf5
from .conftest import TMP_DIR


def test_io_create_and_delete_dir(io_event_dispatcher: EventDispatcher) -> None:
    """Test creation and deletion of (nested) directories."""
    assert not os.path.isdir(TMP_DIR), 'did not expect TMP dir to be present on disk'

    # test creation of TMP dir
    new_dir = create_dir(TMP_DIR, io_event_dispatcher)
    assert new_dir == TMP_DIR, 'expected returned directory to be the same as input directory'
    assert os.path.isdir(TMP_DIR), 'expected TMP dir to be created on disk'

    # test deletion of TMP dir
    delete_dir(TMP_DIR, io_event_dispatcher)
    assert not os.path.isdir(TMP_DIR), 'expected TMP dir to be deleted from disk'

    def fill_dir(directory, depth):
        """Fill the directory with temp files/directories unto the specified depth reaches zero."""
        if depth == 0:
            return

        # create temp file
        file_path = os.path.join(directory, 'file.txt')
        with open(file_path, 'w', encoding='utf-8'):
            print('Creating file', file_path)

        assert os.path.isfile(file_path), 'expected nested file to be created on disk'

        # create temp dir
        dir_path = os.path.join(directory, 'tmp')
        create_dir(dir_path, io_event_dispatcher)
        assert os.path.isdir(dir_path), 'expected nested dir to be created on disk'

        # fill temp dir recursively
        fill_dir(dir_path, depth - 1)

    # test deletion of nested directories/files
    create_dir(TMP_DIR, io_event_dispatcher)
    fill_dir(TMP_DIR, 10)
    delete_dir(TMP_DIR, io_event_dispatcher)

    assert not os.path.isdir(TMP_DIR), 'expected nested TMP dir to be deleted from disk'


@pytest.mark.parametrize('create_file, file_ext, load_file', [
    (create_json, '.json', load_json),
    (create_yml, '.yml', load_yml)
])
def test_io_json_and_yml(
        create_file: Callable[[str, Union[Dict[str, Any], List], EventDispatcher], None],
        file_ext: str,
        load_file: Callable[[str], Union[Dict[str, Any], List]],
        io_tmp_dir: str,
        io_event_dispatcher: EventDispatcher) -> None:
    """Test the save/load functions from yml and json with various file contents."""
    file_list = [None, True, False, 0, 1.0, 'a', [], {}]
    file_dict = {}
    for val in file_list:
        file_dict[str(val)] = val

    for file_data in [file_dict, file_list]:
        file_path = os.path.join(io_tmp_dir, 'test_file' + file_ext)

        create_file(file_path, file_data, io_event_dispatcher)
        assert os.path.isfile(file_path), 'expected file to be saved to disk'

        loaded_file_data = load_file(file_path)
        assert isinstance(loaded_file_data, type(file_data)), \
            'expected loaded data from disk to be the same type after saving and loading'
        assert len(loaded_file_data) == len(file_data), \
            'expected loaded data length to be the same after saving and loading'

        assert loaded_file_data == file_data, \
            'expected file contents to be the same after saving and loading'

        delete_file(file_path, io_event_dispatcher)


def test_io_hdf5_array(io_tmp_dir: str) -> None:
    """Test the save/load hdf5 array functions with integer and float arrays."""
    int_array = list(range(0, 10))
    float_array = []
    for i in int_array:
        float_array.append(float(i))

    for array in [int_array, float_array]:
        array_name = 'array_name'
        array_file_path = os.path.join(io_tmp_dir, 'test_file.hdf5')

        save_array_to_hdf5(array_file_path, array, array_name)
        assert os.path.isfile(array_file_path), 'expected file to be saved to disk'

        loaded_array = load_array_from_hdf5(array_file_path, array_name)
        assert isinstance(loaded_array, np.ndarray), 'expected numpy array to be loaded from disk'
        for i, array_val in enumerate(array):
            assert loaded_array[i] == array_val, \
                'expected array contents to be the same after saving and loading'
