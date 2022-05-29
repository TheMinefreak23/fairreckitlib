"""This module contains event args and a print function for a filter event.

Classes:

    FilterDataframeEventArgs: event args related to filtering a dataframe.

Functions:

    print_filter_event_args: print filter event arguments.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, List

from ...core.events.event_dispatcher import EventArgs


@dataclass
class FilterDataframeEventArgs(EventArgs):
    """Filter Dataframe Event Arguments.

    message: the message describing the parsing failure.
    prefilters: list of filters that is used on the dataframe.
    """

    prefilters: List[Any]


def print_filter_event_args(event_args: FilterDataframeEventArgs, elapsed_time: float=None) -> None:
    """Print filter dataframe event arguments.

    It is assumed that the event started when elapsed_time is None and is finished otherwise.

    Args:
        event_args: the arguments to print.
        elapsed_time: the time that has passed since the filtering started, expressed in seconds.
    """
    if elapsed_time is None:
        print('Filtering dataframe:', event_args.prefilters)
    else:
        print(f'Filtered dataframe in {elapsed_time:1.4f}s')
