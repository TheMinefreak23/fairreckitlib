"""This module contains all event ids and callback functions that are error related.

Constants:

    ON_FAILURE_ERROR: id of the event that is used when a failure occurs.
    ON_RAISE_ERROR: id of the event that is used when an error was raised.

Functions:

    on_error: call when a failure happens or an error was raised.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, List, Tuple

ON_FAILURE_ERROR = 'Error.on_failure'
ON_RAISE_ERROR = 'Error.on_raise'


def get_error_events() -> List[Tuple[str, Callable[[Any], None]]]:
    """Get all error events.

    The Call backs are specified below and serve as a default
    implementation for the RecommenderSystem class including the keyword arguments
    that are available.

    Returns:
        a list of pairs in the format (event_id, func_on_event)
    """
    return [
        (ON_FAILURE_ERROR, on_error),
        (ON_RAISE_ERROR, on_error)
    ]


def on_error(event_listener: Any, **kwargs) -> None:
    """Call back when an error has occurred.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        msg(str): the message describing the error.
    """
    if event_listener.verbose:
        print(kwargs['msg'])
