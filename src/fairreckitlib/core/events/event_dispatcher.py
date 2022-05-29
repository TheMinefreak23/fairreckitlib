"""This module contains a class that implements dispatcher/listener behaviour.

Classes:

    EventDispatcher: can dispatch an event to the respectively subscribed listeners.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, Optional, Tuple

from .event_args import EventArgs


class EventDispatcher:
    """Event Dispatcher provides tools to communicate events to listeners.

    The dispatcher is a centralized object that keeps track of all registered listeners
    for a specified event ID. These listeners can be added and removed dynamically.

    Public methods:

    add_listener
    get_num_listeners
    get_num_listeners_total
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
            func_on_event: Tuple[
                Callable[[Any, EventArgs], None],
                Optional[Callable[[Any, EventArgs], None]]
            ]) -> None:
        """Add a listener for the specified event ID.

        The event_listener and func_on_event are joined as a tuple to describe the listener
        and is expected to be unique. The listener will be notified by the dispatcher
        every time that the event arguments with the specified ID is propagated.
        This function raises a KeyError when the listener tuple already exists.

        Args:
            event_id: unique ID that classifies the event.
            event_listener: the listener of the event. This object is passed to
                the event callback function as the first argument when the event is dispatched.
            func_on_event: the callback functions that are called when the event
                is dispatched. The first argument is the event_listener, followed by event args.
                and any keyword args
        """
        if event_id not in self.listeners:
            self.listeners[event_id] = []

        listener = (event_listener, func_on_event)
        if listener in self.listeners[event_id]:
            raise KeyError('listener is already registered for event', event_id)

        self.listeners[event_id].append(listener)

    def get_num_listeners(self, event_id: str) -> Optional[int]:
        """Get the amount of listeners for the specified event id.

        Args:
            event_id: the event id to query the number of listeners of.

        Returns:
            the number of listeners of the event or None when the event is not registered.
        """
        return len(self.listeners[event_id]) if event_id in self.listeners else None

    def get_num_listeners_total(self) -> int:
        """Get the total amount of listeners for the dispatcher.

        Returns:
            the number of listeners in total.
        """
        num_listeners = 0

        for event_id, _ in self.listeners.items():
            num_listeners += self.get_num_listeners(event_id)

        return num_listeners

    def remove_listener(
            self,
            event_id: str,
            event_listener: Any,
            func_on_event: Tuple[
                Callable[[Any, EventArgs], None],
                Optional[Callable[[Any, EventArgs], None]]
            ]) -> None:
        """Remove a listener for the specified event ID.

        The event_listener and func_on_event are joined as a tuple to describe the listener
        and is expected to be unique. The listener is also expected to be identical to the one
        that was used in 'add_listener' and will no longer be notified by the dispatcher when the
        event arguments with the specified ID is propagated.
        This function raises a KeyError when either the event_id or listener tuple does not exist.

        Args:
            event_id: unique ID that classifies the event.
            event_listener: the listener of the event. This object is passed to
                the event callback function as the first argument when the event is dispatched.
            func_on_event: the callback functions that are called when the event
                is dispatched. The first argument is the event_listener, followed by event args
                and any keyword args.
        """
        if event_id not in self.listeners:
            raise KeyError('event is not registered:', event_id)

        listener = (event_listener, func_on_event)
        if listener not in self.listeners[event_id]:
            raise KeyError('listener is not registered for event:', event_id)

        self.listeners[event_id].remove(listener)

    def dispatch(self, event_args: EventArgs, **kwargs) -> bool:
        """Dispatch event arguments with the corresponding event ID.

        The event arguments with the specified ID will be propagated to all registered listeners.
        For each listener their respective callback function is called with the listener
        as the first argument, followed by the specified event args and keyword args.

        Args:
            event_args: the event's arguments.

        Keyword Args:
            Any: varies depending on the event.

        Returns:
            whether the event was dispatched to any registered listeners.
        """
        if event_args.event_id not in self.listeners:
            return False

        for (event_listener, func_on_event) in self.listeners[event_args.event_id]:
            internal_func, external_func = func_on_event
            internal_func(event_listener, event_args, **kwargs)
            if external_func is not None:
                external_func(event_listener, event_args, **kwargs)

        return True
