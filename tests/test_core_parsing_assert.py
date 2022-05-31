"""This module tests the assertion parsing functionality.

Functions:

    test_parse_assert_empty_container: test assertion of (non) empty containers.
    test_parse_assert_key_in_dict_and_one_of_list: test assertion of key/value in container.
    test_parse_assert_is_type: test assertion for value to be of a certain type.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Dict, List, Union

import pytest

from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.parsing.parse_assert import \
    assert_is_container_not_empty, assert_is_key_in_dict, assert_is_one_of_list, assert_is_type


@pytest.mark.parametrize('container', [{}, [], {'0':0}, [0]])
def test_parse_assert_empty_container(container: Union[Dict, List]) -> None:
    """Test the assertion for (non) empty containers.

    Args:
        container: the container to assert
    """
    event_dispatcher = EventDispatcher()

    if len(container) == 0:
        assert not assert_is_container_not_empty(container, event_dispatcher, ''), \
            'expected container to be empty'
    else:
        assert assert_is_container_not_empty(container, event_dispatcher, ''), \
            'expected container to have entries'


def test_parse_assert_key_in_dict_and_one_of_list() -> None:
    """Test the assertion of a key present in a dictionary or a value present in a list."""
    event_dispatcher = EventDispatcher()

    for i in list(range(0, 100)):
        # keys are assumed to be strings, value can be Any
        key = str(i)
        assert assert_is_key_in_dict(key, {key:i}, event_dispatcher, ''), \
            'expected key to be in the dictionary.'
        assert not assert_is_key_in_dict(key, {}, event_dispatcher, ''), \
            'did not expect key in the dictionary.'

        assert assert_is_one_of_list(i, [i], event_dispatcher, ''), \
            'expected key to be in the list.'
        assert not assert_is_one_of_list(i, [], event_dispatcher, ''), \
            'did not expect key to be in the list.'


def test_parse_assert_is_type() -> None:
    """Test the assertion of various primitive types and containers."""
    event_dispatcher = EventDispatcher()
    types = [None, True, False, 0, 0.0, 'a', [], {'dict':0}, {'set'}]
    for expected_type in types:
        expected_type = type(expected_type)

        for value in types:
            if isinstance(value, expected_type):
                assert assert_is_type(value, expected_type, event_dispatcher, ''), \
                    'expected types to be the same.'
            else:
                assert not assert_is_type(value, expected_type, event_dispatcher, ''), \
                    'expected types to be different.'
