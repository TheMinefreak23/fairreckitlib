"""This module contains event args and a print function for a rating conversion event.

Classes:

    SplitDataframeEventArgs: event args related to splitting dataframes.

Functions:

    print_split_event_args: print dataframe split event arguments.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

from ...core.events.event_dispatcher import EventArgs
from .split_config import SplitConfig


@dataclass
class SplitDataframeEventArgs(EventArgs):
    """Split dataframe Event Arguments.

    event_id: the unique ID that classifies the splitting event.
    split_config: the splitting configuration that is used.
    """

    split_config: SplitConfig


def print_split_event_args(event_args: SplitDataframeEventArgs, elapsed_time: float=None) -> None:
    """Print split dataframe event arguments.

    It is assumed that the event started when elapsed_time is None and is finished otherwise.

    Args:
        event_args: the arguments to print.
        elapsed_time: the time that has passed since the splitting started, expressed in seconds.
    """
    if elapsed_time is None:
        print('Splitting dataframe:', event_args.split_config.get_split_ratio_string(),
              '=>', event_args.split_config.name)
    else:
        print(f'Split dataframe in {elapsed_time:1.4f}s')
