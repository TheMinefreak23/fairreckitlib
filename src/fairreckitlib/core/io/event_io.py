"""This module contains all event ids, event args and a print switch that are IO related.

Constants:

    ON_MAKE_DIR: id of the event that is used when a directory is created.
    ON_REMOVE_DIR: id of the event that is used when a directory is removed.
    ON_CREATE_FILE: id of the event that is used when a file is created.
    ON_REMOVE_FILE: id of the event that is used when a file is removed.
    ON_RENAME_FILE: id of the event that is used when a file is renamed.

Classes:

    DirEventArgs: event args related to a directory.
    FileEventArgs: event args related to a file.
    DataframeEventArgs: event args related to a dataframe.
    RenameFileEventArgs: event args related to renaming a file.

Functions:

    get_io_events: list of IO event IDs.
    get_io_event_print_switch: switch to print IO event arguments by ID.
    print_load_dataframe_event_args: print dataframe event arguments for loading.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from ..events.event_dispatcher import EventArgs

ON_MAKE_DIR = 'IO.on_make_dir'
ON_REMOVE_DIR = 'IO.on_remove_dir'

ON_CREATE_FILE = 'IO.on_create_file'
ON_REMOVE_FILE = 'IO.on_remove_file'


@dataclass
class DirEventArgs(EventArgs):
    """Directory Event Arguments.

    event_id: the unique ID that classifies the directory event.
    directory: the path to the directory.
    """

    directory: str


@dataclass
class FileEventArgs(EventArgs):
    """File Event Arguments.

    event_id: the unique ID that classifies the file event.
    file_path: the path to the file.
    """

    file_path: str


@dataclass
class DataframeEventArgs(FileEventArgs):
    """Dataframe Event Arguments.

    event_id: the unique ID that classifies the dataframe event.
    file_path: the path to the dataframe file.
    dataframe_name: the name of the dataframe.
    """

    dataframe_name: str


def get_io_events() -> List[str]:
    """Get a list of IO event IDs.

    Returns:
        a list of unique IO event IDs.
    """
    return [
        # DirEventArgs
        ON_MAKE_DIR,
        ON_REMOVE_DIR,
        # FileEventArgs
        ON_CREATE_FILE,
        ON_REMOVE_FILE,
    ]


def get_io_event_print_switch() -> Dict[str, Callable[[EventArgs], None]]:
    """Get a switch that prints IO event IDs.

    Returns:
        the print IO event switch.
    """
    return {
        ON_MAKE_DIR:
            lambda args: print('Creating directory:', args.directory),
        ON_REMOVE_DIR:
            lambda args: print('Removing directory:', args.directory),
        ON_CREATE_FILE:
            lambda args: print('Creating file:', args.file_path),
        ON_REMOVE_FILE:
            lambda args: print('Removing file:', args.file_path),
    }


def print_load_df_event_args(event_args: DataframeEventArgs, elapsed_time: float=None)-> None:
    """Print dataframe event arguments for loading.

    It is assumed that the event started when elapsed_time is None and is finished otherwise.

    Args:
        event_args: the arguments to print.
        elapsed_time: the time that has passed since the loading started, expressed in seconds.
    """
    if elapsed_time is None:
        print('Loading', event_args.dataframe_name, 'from', event_args.file_path)
    else:
        print('Loaded', event_args.dataframe_name, f'in {elapsed_time:1.4f}s')
