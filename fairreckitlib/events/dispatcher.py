"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


class EventDispatcher:
    """Event Dispatcher provides tools to communicate events to listeners.

    The dispatcher is a centralized object that keeps track of all registered listeners
    for a specified event ID. These listeners can be added and removed dynamically.
    """

    def __init__(self):
        self.__listeners = {}

    def add_listener(self, event_id, event_listener, func_on_event):
        """Adds a listener for the specified event ID.

        The event_listener and func_on_event are joined as a tuple to
        describe the listener. The listener will be notified by the dispatcher
        every time that the event with the specified ID is propagated.

        Args:
            event_id: unique ID that classifies the event.
            event_listener(object): the listener of the event. This object is passed to
                the event callback function as the first argument when the event is dispatched.
            func_on_event(function): the callback function that is called when the event
                is dispatched. The first argument is the event_listener, followed by keyword args.
        """
        if not event_id in self.__listeners:
            self.__listeners[event_id] = []

        listener = (event_listener, func_on_event)
        self.__listeners[event_id].append(listener)

    def remove_listener(self, event_id, event_listener, func_on_event):
        """Removes a listener for the specified event ID.

        The event_listener and func_on_event are joined as a tuple to
        describe the listener. The listener is expected to be identical to the
        one that was used in 'add_listener' and will no longer be notified by the
        dispatcher when the event with the specified ID is propagated.

        Args:
            event_id: unique ID that classifies the event.
            event_listener(object): the listener of the event. This object is passed to
                the event callback function as the first argument when the event is dispatched.
            func_on_event(function): the callback function that is called when the event
                is dispatched. The first argument is the event_listener, followed by keyword args.
        """
        if not event_id in self.__listeners:
            return

        listener = (event_listener, func_on_event)
        if not listener in self.__listeners[event_id]:
            return

        self.__listeners[event_id].remove(listener)

    def dispatch(self, event_id, **kwargs):
        """Dispatch an event with the specified ID.

        The event with the specified ID will be propagated to all registered listeners.
        For each listener their respective callback function is called with the listener
        as the first argument, followed by the specified keyword arguments.

        Args:
            event_id: unique ID that classifies the event.

        Keyword Args:
            kwargs: varies depending on the event.
        """
        if not event_id in self.__listeners:
            return

        for _, (event_listener, func_on_event) in enumerate(self.__listeners[event_id]):
            func_on_event(event_listener, **kwargs)
