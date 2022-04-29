"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.events import config_event


def is_container_not_empty(src_container, event_dispatcher, error_msg, **kwargs):
    """Asserts whether the specified container is not empty.

    Args:
        src_container(container): the container to assert.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.
        error_msg(str): the error message describing the assertion failure.

    Keyword Args:
        one_of_list(array like): list of values that contains the expected value.
        expect(object): the type that is expected to be parsed.
        actual(object): the type that is attempted to be parsed.
        default(object): the default value that is returned after parsing.

    Returns:
        bool: whether the assertion has passed.
    """
    if len(src_container) == 0:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg=error_msg,
            **kwargs
        )
        return False

    return True


def is_key_in_dict(src_key, src_dict, event_dispatcher, error_msg, **kwargs):
    """Asserts whether the specified key is present in the specified dictionary.

    Args:
        src_key(object): the key to assert.
        src_dict(dict): the dictionary to check the key in.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.
        error_msg(str): the error message describing the assertion failure.

    Keyword Args:
        one_of_list(array like): list of values that contains the expected value.
        expect(object): the type that is expected to be parsed.
        actual(object): the type that is attempted to be parsed.
        default(object): the default value that is returned after parsing.

    Returns:
        bool: whether the assertion has passed.
    """
    if not src_key in src_dict:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg=error_msg,
            **kwargs
        )
        return False

    return True


def is_one_of_list(src_value, src_list, event_dispatcher, error_msg, **kwargs):
    """Asserts whether the specified value is present in the specified list.

    Args:
        src_value(object): the value to assert.
        src_list(array like): the list to check the value in.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.
        error_msg(str): the error message describing the assertion failure.

    Keyword Args:
        expect(object): the type that is expected to be parsed.
        actual(object): the type that is attempted to be parsed.
        default(object): the default value that is returned after parsing.

    Returns:
        bool: whether the assertion has passed.
    """
    if not src_value in src_list:
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg=error_msg,
            one_of_list=src_list,
            **kwargs
        )
        return False

    return True


def is_type(value, expected_type, event_dispatcher, error_msg, **kwargs):
    """Asserts whether the specified value is of the expected type.

    Args:
        value(object): the value to assert.
        expected_type(dict): the type that is expected.
        event_dispatcher(EventDispatcher): to dispatch the parse event on failure.
        error_msg(str): the error message describing the assertion failure.

    Keyword Args:
        one_of_list(array like): list of values that contains the expected value.
        default(object): the default value that is returned after parsing.

    Returns:
        bool: whether the assertion has passed.
    """
    if not isinstance(value, expected_type):
        event_dispatcher.dispatch(
            config_event.ON_PARSE,
            msg=error_msg,
            expect=expected_type,
            actual=type(value),
            **kwargs
        )
        return False

    return True
