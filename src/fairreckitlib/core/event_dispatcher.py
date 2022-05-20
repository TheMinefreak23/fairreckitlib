"""This module contains a class that implements dispatcher/listener behaviour.

Classes:

    EventDispatcher: can dispatch an event to the respectively subscribed listeners.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, Optional, Tuple


class EventDispatcher:
    """Event Dispatcher provides tools to communicate events to listeners.

    The dispatcher is a centralized object that keeps track of all registered listeners
    for a specified event ID. These listeners can be added and removed dynamically.

    Public methods:

    add_listener
    remove_listener
    dispatch
    """

    def __init__(self):
        """Construct the EventDispatcher."""
        self.listeners = {}

    def add_listener(
            self,
            event_id: str,
            event_listener: Any,
            func_on_event: Tuple[Callable[[Any], None], Optional[Callable[[Any], None]]]) -> None:
        """Add a listener for the specified event ID.

        The event_listener and func_on_event are joined as a tuple to
        describe the listener. The listener will be notified by the dispatcher
        every time that the event with the specified ID is propagated.

        Args:
            event_id: unique ID that classifies the event.
            event_listener: the listener of the event. This object is passed to
                the event callback function as the first argument when the event is dispatched.
            func_on_event: the callback functions that are called when the event
                is dispatched. The first argument is the event_listener, followed by keyword args.
        """
        if event_id not in self.listeners:
            self.listeners[event_id] = []

        listener = (event_listener, func_on_event)
        self.listeners[event_id].append(listener)

    def remove_listener(
            self,
            event_id: str,
            event_listener: Any,
            func_on_event: Tuple[Callable[[Any], None], Optional[Callable[[Any], None]]]) -> None:
        """Remove a listener for the specified event ID.

        The event_listener and func_on_event are joined as a tuple to
        describe the listener. The listener is expected to be identical to the
        one that was used in 'add_listener' and will no longer be notified by the
        dispatcher when the event with the specified ID is propagated.

        Args:
            event_id: unique ID that classifies the event.
            event_listener: the listener of the event. This object is passed to
                the event callback function as the first argument when the event is dispatched.
            func_on_event: the callback functions that are called when the event
                is dispatched. The first argument is the event_listener, followed by keyword args.
        """
        if event_id not in self.listeners:
            return

        listener = (event_listener, func_on_event)
        if listener not in self.listeners[event_id]:
            return

        self.listeners[event_id].remove(listener)

    def dispatch(self, event_id: str, **kwargs) -> None:
        """Dispatch an event with the specified ID.

        The event with the specified ID will be propagated to all registered listeners.
        For each listener their respective callback function is called with the listener
        as the first argument, followed by the specified keyword arguments.

        Args:
            event_id: unique ID that classifies the event.

        Keyword Args:
            kwargs: varies depending on the event.
        """
        if event_id not in self.listeners:
            return

        for _, (event_listener, func_on_event) in enumerate(self.listeners[event_id]):
            internal_func, external_func = func_on_event
            internal_func(event_listener, **kwargs)
            if external_func is not None:
                external_func(event_listener, **kwargs)
