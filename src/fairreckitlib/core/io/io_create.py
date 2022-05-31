"""This module contains IO functions that create something on the disk and dispatch an IO event.

Functions:

    create_dir: create a directory on the disk with IO event dispatching.
    create_json: create a json file on the disk with IO event dispatching.
    create_yml: create a yml file on the disk with event dispatching.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from typing import Dict

from ..events.event_dispatcher import EventDispatcher
from .event_io import ON_CREATE_FILE, ON_MAKE_DIR, DirEventArgs, FileEventArgs
from .io_utility import save_json, save_yml


def create_dir(directory: str, event_dispatcher: EventDispatcher) -> str:
    """Create the specified directory.

    This functions checks whether the directory exists and the
    event is only dispatched when the directory did not exist yet.

    Args:
        directory: the directory to create on the disk.
        event_dispatcher: used to dispatch the IO event.

    Returns:
        the directory path.
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)
        event_dispatcher.dispatch(DirEventArgs(ON_MAKE_DIR, directory))

    return directory


def create_json(
        file_path: str,
        data: Dict,
        event_dispatcher: EventDispatcher,
        *,
        encoding: str='utf-8',
        indent=1) -> None:
    """Create a JSON file with the specified data.

    The IO event is dispatched after the file is created.

    Args:
        file_path: path to where the json file will be stored.
        data: the source dictionary to save in the file.
        event_dispatcher: used to dispatch the IO event.
        encoding: the encoding to use for writing the file contents.
        indent: the indent level for pretty printing JSON array elements and object members.
    """
    save_json(file_path, data, encoding=encoding, indent=indent)
    event_dispatcher.dispatch(FileEventArgs(ON_CREATE_FILE, file_path))


def create_yml(
        file_path: str,
        data: Dict,
        event_dispatcher: EventDispatcher,
        *,
        encoding: str='utf-8') -> None:
    """Create a YML file with the specified data.

    The IO event is dispatched after the file is created.

    Args:
        file_path: path to where the json file will be stored.
        data: the source dictionary to save in the file.
        event_dispatcher: used to dispatch the IO event.
        encoding: the encoding to use for writing the file contents.
    """
    save_yml(file_path, data, encoding=encoding)
    event_dispatcher.dispatch(FileEventArgs(ON_CREATE_FILE, file_path))
