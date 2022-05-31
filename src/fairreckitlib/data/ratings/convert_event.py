"""This module contains event args and a print function for a rating conversion event.

Classes:

    ConvertRatingsEventArgs: event args related to converting dataframe ratings.

Functions:

    print_convert_event_args: print rating conversion event arguments.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

from ...core.events.event_dispatcher import EventArgs
from .convert_config import ConvertConfig


@dataclass
class ConvertRatingsEventArgs(EventArgs):
    """Convert Ratings Event Arguments.

    event_id: the unique ID that classifies the rating conversion event.
    convert_config: the rating conversion configuration that is used.
    """

    convert_config: ConvertConfig


def print_convert_event_args(event_args: ConvertRatingsEventArgs, elapsed_time: float=None) -> None:
    """Print convert ratings event arguments.

    It is assumed that the event started when elapsed_time is None and is finished otherwise.

    Args:
        event_args: the arguments to print.
        elapsed_time: the time that has passed since the conversion started, expressed in seconds.
    """
    if elapsed_time is None:
        print('Converting dataframe ratings:', event_args.convert_config.name)
    else:
        print(f'Converted dataframe ratings in {elapsed_time:1.4f}s')
