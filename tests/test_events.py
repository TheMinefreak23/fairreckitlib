"""This module tests the event dispatching/listening functionality.

Classes:

    AssertEventArgs: event args to be used for testing assertions.

Functions:

    test_event_dispatcher_add_and_remove: test add and remove listener functionality.
    test_event_dispatch: test event dispatch functionality.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass

import pytest

from src.fairreckitlib.core.events.event_args import EventArgs
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher

# ids are unique
event_id1, event_id2 = '1', '2'
# dummy event callback that does nothing
on_nothing = lambda listener, args: None


@dataclass
class AssertEventArgs(EventArgs):
    """Assert Event Args.

    event_id: the unique ID that classifies the event.
    test: to test assertions.
    """

    test: bool = True


def test_event_dispatcher_add_and_remove() -> None:
    """Test the add and remove listener functionality of the event dispatcher."""
    event_dispatcher = EventDispatcher()
    # listeners are unique
    event_listener1, event_listener2 = int(event_id1), int(event_id2)
    # callback can be shared
    func_on_event = (on_nothing, None)

    assert event_dispatcher.get_num_listeners_total() == 0, \
        'did not expect any registered event listeners at all.'
    assert event_dispatcher.get_num_listeners(event_id1) is None, \
        'did not expect any registered listeners for an unknown event id.'

    # test failure for an unknown event id
    pytest.raises(KeyError, event_dispatcher.remove_listener,
                  event_id1, event_listener1, func_on_event)

    event_dispatcher.add_listener(event_id1, True, func_on_event)

    # test failure for an unknown event listener
    pytest.raises(KeyError, event_dispatcher.remove_listener,
                  event_id1, event_listener2, func_on_event)

    assert event_dispatcher.get_num_listeners_total() == 1, \
        'expected only one listener in total.'
    assert event_dispatcher.get_num_listeners(event_id1) == 1, \
        'expected only one listener for the event.'

    # test failure for an already known event listener
    pytest.raises(KeyError, event_dispatcher.add_listener,
                  event_id1, event_listener1, func_on_event)

    assert event_dispatcher.get_num_listeners_total() == 1, \
        'expected only one listener in total after failing to add a duplicate listener.'
    assert event_dispatcher.get_num_listeners(event_id1) == 1, \
        'expected only one listener for the event after failing to add a duplicate listener.'

    event_dispatcher.add_listener(event_id1, event_listener2, func_on_event)
    assert event_dispatcher.get_num_listeners_total() == 2, \
        'expected two listeners in total.'
    assert event_dispatcher.get_num_listeners(event_id1) == 2, \
        'expected two unique listeners for the event.'

    event_dispatcher.add_listener(event_id2, event_listener2, func_on_event)
    assert event_dispatcher.get_num_listeners_total() == 3, \
        'expected three listeners in total.'
    assert event_dispatcher.get_num_listeners(event_id1) == 2, \
        'did not expect the number of listeners to change.'

    event_dispatcher.remove_listener(event_id1, event_listener1, func_on_event)
    assert event_dispatcher.get_num_listeners(event_id1) == 1, \
        'expected one listener less for the event after removal.'


def test_event_dispatch() -> None:
    """Test the dispatch functionality of the event dispatcher."""
    elapsed_time = 5.0
    def on_ext_event(listener: bool, event_args: EventArgs) -> None:
        """Test external event callback."""
        assert not listener, 'expected listener to be False'

        if isinstance(event_args, AssertEventArgs):
            assert not event_args.test, \
                'expected assert event args to be the same as input.'

    def on_int_event(listener: bool, event_args: EventArgs, **kwargs) -> None:
        """Test internal event callback."""
        assert listener, 'expected listener to be True.'
        assert event_args.event_id == event_id1

        if isinstance(event_args, AssertEventArgs):
            assert event_args.test, \
                'expected assert event args to be the same as input.'

        if kwargs.get('elapsed_time', False):
            assert kwargs['elapsed_time'] == elapsed_time, \
                'expected keyword argument elapsed_time to be the same as input.'

    event_dispatcher = EventDispatcher()

    # test internal event callback
    event_dispatcher.add_listener(event_id1, True, (on_int_event, None))
    assert event_dispatcher.dispatch(EventArgs(event_id1)), \
        'expected event dispatch to succeed.'
    assert event_dispatcher.dispatch(AssertEventArgs(event_id1)), \
        'expected event dispatch to succeed.'
    assert event_dispatcher.dispatch(AssertEventArgs(event_id1), elapsed_time=elapsed_time), \
        'expected event dispatch to succeed.'
    event_dispatcher.remove_listener(event_id1, True, (on_int_event, None))

    # test external event callback
    event_dispatcher.add_listener(event_id1, False, (on_nothing, on_ext_event))
    assert event_dispatcher.dispatch(EventArgs(event_id1)), \
        'expected event dispatch to succeed.'
    assert event_dispatcher.dispatch(AssertEventArgs(event_id1, False)), \
        'expected event dispatch to succeed.'

    # test dispatch for unknown event id
    assert not event_dispatcher.dispatch(EventArgs(event_id2)), \
        'expected event dispatch to fail.'
    assert not event_dispatcher.dispatch(AssertEventArgs(event_id2)), \
        'expected event dispatch to fail.'
