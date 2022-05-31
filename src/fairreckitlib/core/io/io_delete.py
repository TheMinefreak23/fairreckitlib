"""This module contains IO functions that delete something on the disk and dispatch an IO event.

Functions:

    delete_dir: delete a directory, recursively, with IO event dispatching.
    delete_file: delete a file from the disk with IO event dispatching.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from ..events.event_dispatcher import EventDispatcher
from .event_io import ON_REMOVE_DIR, ON_REMOVE_FILE, DirEventArgs, FileEventArgs


def delete_dir(directory: str, event_dispatcher: EventDispatcher) -> None:
    """Delete the specified directory.

    This functions removes all the files and directories that are present
    in the specified directory path (recursively).
    The IO event is dispatched after the directory and/or files are deleted.

    Args:
        directory: the path to the directory to delete.
        event_dispatcher: used to dispatch the IO event.
    """
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            delete_dir(file_path, event_dispatcher)
        else:
            delete_file(file_path, event_dispatcher)

    os.rmdir(directory)
    event_dispatcher.dispatch(DirEventArgs(ON_REMOVE_DIR, directory))


def delete_file(file_path: str, event_dispatcher: EventDispatcher) -> None:
    """Delete the specified file.

    The IO event is dispatched after the file is deleted.

    Args:
        file_path: the path to the file to delete.
        event_dispatcher: used to dispatch the IO event.
    """
    os.remove(file_path)
    event_dispatcher.dispatch(FileEventArgs(ON_REMOVE_FILE, file_path))
