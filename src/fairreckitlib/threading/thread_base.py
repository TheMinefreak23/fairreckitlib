"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from threading import Thread

from src.fairreckitlib.events.dispatcher import EventDispatcher


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

    Args:
        name(str) the name of the thread.
        events(dict): events to dispatch for this thread

    Keyword Args:
        varying: these are passed to the run function from step 2.
    """
    def __init__(self, name, events, verbose, **kwargs):
        self.__is_running = False
        self.__thread = Thread(target=self.__main, name=name, kwargs=kwargs)
        self.__on_terminate = None

        self.verbose = verbose
        self.event_dispatcher = EventDispatcher()
        for (event_id, func_on_event) in events.items():
            self.event_dispatcher.add_listener(event_id, self, func_on_event)

    def start(self, func_on_terminate):
        """Starts running a thread.

        Args:
            func_on_terminate(function): callback function that is called once
                the thread is finished running and is terminated. This function has
                one argument which is the thread itself.
        """
        if self.__is_running:
            return

        self.__is_running = True
        self.__on_terminate = func_on_terminate
        self.__thread.start()

    def stop(self):
        """Stops running a thread.

        Does not cancel the thread, but rather requests for the thread to finish
        by settings the is_running flag to False. Derived classes need to account
        for this request by checking the status regularly.
        """
        self.__is_running = False

    def get_name(self):
        """Gets the name of the thread.

        Returns:
            name(str): the thread's name.
        """
        return self.__thread.name

    def is_running(self):
        """Returns if the thread is still running.

        Returns:
            is_running(bool): whether the thread is running.
        """
        return self.__is_running

    def on_initialize(self):
        """Initializes the thread.

        This function is called once when the thread is started.
        It should not be used directly, add specific logic in derived classes.
        """

    @abstractmethod
    def on_run(self, **kwargs):
        """Runs the thread.

        This function is called once after the thread is initialized.
        It should not be used directly, add specific logic in derived classes.

        Keyword Args:
            varying: these are passed directly from the thread's constructor.
        """
        raise NotImplementedError()

    def on_terminate(self):
        """Terminates the thread.

        This function is called once after the thread is done running.
        It should not be used directly, add specific logic in derived classes.
        """
        self.__on_terminate(self)

    def __main(self, **kwargs):
        """The main function target of the thread."""
        self.on_initialize()
        self.on_run(**kwargs)
        self.on_terminate()
