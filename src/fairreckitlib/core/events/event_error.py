"""This module contains all event ids, event args and a print switch that are error related.

Constants:

    ON_FAILURE_ERROR: id of the event that is used when a failure occurs.
    ON_RAISE_ERROR: id of the event that is used when an error was raised.

Classes:

    ErrorEventArgs: event args for errors.

Functions:

    get_error_events: list of error event IDs.
    get_error_event_print_switch: switch to print error event arguments by ID.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from .event_args import EventArgs, MessageEventArgs

ON_FAILURE_ERROR = 'Error.on_failure'
ON_RAISE_ERROR = 'Error.on_raise'


@dataclass
class ErrorEventArgs(MessageEventArgs):
    """Error Event Arguments.

    event_id: the unique ID that classifies the error event.
    message: the error message.
    """


def get_error_events() -> List[str]:
    """Get a list of error event IDs.

    Returns:
        a list of unique error event IDs.
    """
    return [
        # ErrorEventArgs
        ON_FAILURE_ERROR,
        ON_RAISE_ERROR
    ]


def get_error_event_print_switch() -> Dict[str, Callable[[EventArgs], None]]:
    """Get a switch that prints error event IDs.

    Returns:
        the print error event switch.
    """
    print_error = lambda args: print(args.message)
    return {
        ON_FAILURE_ERROR: print_error,
        ON_RAISE_ERROR: print_error,
    }
