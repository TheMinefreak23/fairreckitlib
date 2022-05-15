"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Type, Union

from ..event_dispatcher import EventDispatcher
from .parse_event import ON_PARSE


def assert_is_container_not_empty(
        src_container: Union[Dict, List],
        event_dispatcher: EventDispatcher,
        error_msg: str,
        **kwargs) -> bool:
    """Assert whether the specified container is not empty.

    Args:
        src_container: the container to assert.
        event_dispatcher: to dispatch the parse event on failure.
        error_msg: the error message describing the assertion failure.

    Keyword Args:
        one_of_list(List[Any]): list of values that contains the expected value.
        expect(Type): the type that is expected to be parsed.
        actual(Any): the value that is attempted to be parsed.
        default(Any): the default value that is returned after parsing.

    Returns:
        whether the assertion has passed.
    """
    if len(src_container) == 0:
        event_dispatcher.dispatch(
            ON_PARSE,
            msg=error_msg,
            **kwargs
        )
        return False

    return True


def assert_is_key_in_dict(
        src_key: str,
        src_dict: Dict[str, Any],
        event_dispatcher: EventDispatcher,
        error_msg: str,
        **kwargs) -> bool:
    """Assert whether the specified key is present in the specified dictionary.

    Args:
        src_key: the key to assert.
        src_dict: the dictionary to check the key in.
        event_dispatcher: to dispatch the parse event on failure.
        error_msg: the error message describing the assertion failure.

    Keyword Args:
        one_of_list(List[Any]): list of values that contains the expected value.
        expect(Type): the type that is expected to be parsed.
        actual(Any): the value that is attempted to be parsed.
        default(Any): the default value that is returned after parsing.

    Returns:
        whether the assertion has passed.
    """
    if not src_key in src_dict:
        event_dispatcher.dispatch(
            ON_PARSE,
            msg=error_msg,
            **kwargs
        )
        return False

    return True


def assert_is_one_of_list(
        src_value: str,
        src_list: List[str],
        event_dispatcher: EventDispatcher,
        error_msg: str,
        **kwargs) -> bool:
    """Assert whether the specified value is present in the specified list.

    Args:
        src_value: the value to assert.
        src_list: the list to check the value in.
        event_dispatcher: to dispatch the parse event on failure.
        error_msg: the error message describing the assertion failure.

    Keyword Args:
        expect(Type): the type that is expected to be parsed.
        actual(Any): the value that is attempted to be parsed.
        default(str): the default value that is returned after parsing.

    Returns:
        bool: whether the assertion has passed.
    """
    if not src_value in src_list:
        event_dispatcher.dispatch(
            ON_PARSE,
            msg=error_msg,
            one_of_list=src_list,
            **kwargs
        )
        return False

    return True


def assert_is_type(
        value: Any,
        expected_type: Type,
        event_dispatcher: EventDispatcher,
        error_msg: str,
        **kwargs) -> bool:
    """Assert whether the specified value is of the expected type.

    Args:
        value: the value to assert.
        expected_type: the type that is expected.
        event_dispatcher: to dispatch the parse event on failure.
        error_msg: the error message describing the assertion failure.

    Keyword Args:
        one_of_list(List[Any]): list of values that contains the expected value.
        default(Any): the default value that is returned after parsing.

    Returns:
        bool: whether the assertion has passed.
    """
    if not isinstance(value, expected_type):
        event_dispatcher.dispatch(
            ON_PARSE,
            msg=error_msg,
            expect=expected_type,
            actual=type(value),
            **kwargs
        )
        return False

    return True
