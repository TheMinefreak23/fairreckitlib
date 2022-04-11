"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

ON_MAKE_DIR = 'on_make_dir'
ON_REMOVE_DIR = 'on_remove_dir'
ON_REMOVE_FILE = 'on_remove_file'
ON_RENAME_FILE = 'on_rename_file'


def on_make_dir(event_listener, **kwargs):
    """Callback function when a new directory is created.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dir(str): the path of the directory that was created.
    """
    if event_listener.verbose:
        print('Creating directory:', kwargs['dir'])


def on_remove_dir(event_listener, **kwargs):
    """Callback function when an existing directory is removed.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        dir(str): the path of the directory that was removed.
    """
    if event_listener.verbose:
        print('Removing directory:', kwargs['dir'])


def on_remove_file(event_listener, **kwargs):
    """Callback function when an existing file is removed.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        file(str): the path of the file that was removed.
    """
    if event_listener.verbose:
        print('Removing file:', kwargs['file'])


def on_rename_file(event_listener, **kwargs):
    """Callback function when an existing file is renamed.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        src_file(str): the path of the file before renaming.
        dst_file(str): the path of the file after renaming.
    """
    if event_listener.verbose:
        print('Renaming file:', kwargs['src_file'], 'to', kwargs['dst_file'])
