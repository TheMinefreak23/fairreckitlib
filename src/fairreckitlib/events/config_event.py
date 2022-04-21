"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

ON_PARSE = 'on_parse'


def on_parse(event_listener, **kwargs):
    """Callback function when parsing is executed.

    Args:
        event_listener(object): the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        msg(str): the message describing the parsing.
        one_of_list(array like): list of values that contains the expected value.
        expect(object): the type that is expected to be parsed.
        actual(object): the type that is attempted to be parsed.
        default(object): the default value that is returned after parsing.
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
