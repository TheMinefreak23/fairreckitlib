"""This module contains an event id and callback function used during parsing.

Constants:

    ON_PARSE: id of the event that is used when parsing fails.

Functions:

    on_parse: call when parsing fails.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any

ON_PARSE = 'Config.on_parse'


def on_parse(event_listener: Any, **kwargs) -> None:
    """Call back function when parsing is executed and fails.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        msg(str): the message describing the parsing failure.
        one_of_list(List[Any]): list of values that contains the expected value.
        expect(Type): the type that is expected to be parsed.
        actual(Type): the type that is attempted to be parsed.
        default(Any): the default value that is returned after parsing.
    """
    if event_listener.verbose:
        print(kwargs['msg'])
        if kwargs.get('one_of_list') is not None:
            print('\texpected one of: ' + str(kwargs['one_of_list']))
        if kwargs.get('expect') is not None:
            print('\texpect:', kwargs['expect'])
        if kwargs.get('actual') is not None:
            print('\tactual:', kwargs['actual'])
        if kwargs.get('default') is not None:
            print('\tdefault to:', kwargs['default'])
