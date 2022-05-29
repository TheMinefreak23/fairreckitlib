"""This module contains an event id, event args and print function for a parsing event.

Constants:

    ON_PARSE: id of the event that is used when parsing fails.

Classes:

    ParseEventArgs: event args related to parsing.

Functions:

    print_parse_event: print parse event arguments.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, List, Type

from ..events.event_args import MessageEventArgs

ON_PARSE = 'Config.on_parse'


@dataclass
class ParseEventArgs(MessageEventArgs):
    """Parse Event Arguments.

    Only the message is required, other variables are optional.

    message: the message describing the parsing failure.
    one_of_list: a list of values that contains the expected value.
    expected_type: the type that is expected to be parsed.
    actual_type: the type that is attempted to be parsed.
    default_value: the default value that is returned after parsing.
    """

    one_of_list: List[Any] = None
    expected_type: Type = None
    actual_type: Type = None
    default_value: Any = None


def print_parse_event(event_args: ParseEventArgs) -> None:
    """Print parse event arguments.

    Args:
        event_args: the arguments to print.
    """
    print(event_args.message)
    if event_args.one_of_list is not None:
        print('\texpected one of: ' + str(event_args.one_of_list))
    if event_args.expected_type is not None:
        print('\texpect:', event_args.expected_type)
    if event_args.actual_type is not None:
        print('\tactual:', event_args.actual_type)
    if event_args.default_value is not None:
        print('\tdefault to:', event_args.default_value)
