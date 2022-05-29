"""This module contains the base event arguments dataclasses.

Classes:

    EventArgs: base event args for all events.
    MessageEventArgs: event args that has a message.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass


@dataclass
class EventArgs:
    """Base Event Arguments.

    Event arguments classes are allowed to be shared for different event ids.

    event_id: the unique ID that classifies the event.
    """

    event_id: str

@dataclass
class MessageEventArgs(EventArgs):
    """Message Event Arguments.

    event_id: the unique ID that classifies the message event.
    message: the message.
    """

    message: str
