"""This module contains the base class for threads.

Classes:

    ThreadBase: base class that implements basic threading functionality.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from threading import Thread
from typing import Any, Callable

from ..events.event_dispatcher import EventDispatcher


class ThreadBase(metaclass=ABCMeta):
    """Base class for all threads.

    Wraps the threading module in a class interface. The main function of each
    thread consists of three steps:

    1) Initialize
    2) Run
    3) Terminate

    Threads are not running on creation, only after the start() function is called.
    The function is_running() will return True as long as the thread is active,
    once the stop() function is called it requests the thread to finish.
    Threads cannot be cancelled so any derived class logic needs to account for this request
    by checking the is_running function pointer (in step 2) regularly.

    Abstract methods:

    on_initialize (optional)
    on_run (required)
    on_terminate (optional)

    Public methods:

    get_name
    is_running
    start
    stop
    """

    def __init__(self, name: str, verbose: bool, **kwargs):
        """Construct the BaseThread.

        Args:
            name the name of the thread.
            verbose: whether the thread should give verbose output.

        Keyword Args:
            varying: these are passed to the run function from step 2.
        """
        self.running = False
        self.thread = Thread(target=self.main, name=name, kwargs=kwargs)
        self.terminate_callback = None

        self.verbose = verbose
        self.event_dispatcher = EventDispatcher()

    def start(self, terminate_callback: Callable[[Any], None]) -> bool:
        """Start running the thread.

        Args:
            terminate_callback: call back function that is called once the thread is
            finished running and is terminated. This function has one argument
            which is the thread itself.

        Returns:
            True when the thread successfully started or False when the thread is already running.
        """
        if self.running:
            return False

        self.running = True
        self.terminate_callback = terminate_callback
        self.thread.start()

        return True

    def stop(self) -> None:
        """Stop running the thread.

        Does not cancel the thread, but rather requests for the thread to finish
        by settings the is_running flag to False. Derived classes need to account
        for this request by checking the status regularly.
        """
        self.running = False

    def get_name(self) -> str:
        """Get the name of the thread.

        Returns:
            the thread's name.
        """
        return self.thread.name

    def is_running(self) -> bool:
        """Get if the thread is still running.

        Returns:
            whether the thread is running.
        """
        return self.running

    def on_initialize(self) -> None:
        """Initialize the thread.

        This function is called once when the thread is started.
        It should not be used directly, add specific logic in derived classes.
        """

    @abstractmethod
    def on_run(self, **kwargs) -> None:
        """Run the thread.

        This function is called once after the thread is initialized.
        It should not be used directly, add specific logic in derived classes.

        Keyword Args:
            varying: these are passed directly from the thread's constructor.
        """
        raise NotImplementedError()

    def on_terminate(self) -> None:
        """Terminate the thread.

        This function is called once after the thread is done running.
        It should not be used directly, add specific logic in derived classes.
        Moreover, derived classes are expected to call their super implementation
        after their own implementation is finished.
        """
        self.terminate_callback(self)

    def main(self, **kwargs) -> None:
        """Run the main function target of the thread.

        Keyword Args:
            varying: these are passed directly from the thread's constructor.
        """
        self.on_initialize()
        self.on_run(**kwargs)
        self.on_terminate()
