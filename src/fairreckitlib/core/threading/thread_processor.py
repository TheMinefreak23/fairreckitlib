"""This module contains a processor that handles active threads.

Classes:

    ThreadProcessor: class that starts new and stops running threads.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import List

from .thread_base import ThreadBase


class ThreadProcessor:
    """Processor for multiple threads (derived from ThreadBase class).

    Keeps track of all threads that are started by this processor.
    The processor acquires ownership of these aforementioned threads and
    will dispose of them after they are finished.
    Additionally, these threads can be stopped as well.

    Public methods:

    get_active_threads
    get_num_active
    is_active_thread
    start
    stop
    """

    def __init__(self):
        """Construct the ThreadProcessor."""
        self.threads = {}

    def get_active_threads(self) -> List[str]:
        """Get the names of any active threads.

        Returns:
            a list of thread names that are currently running.
        """
        active_threads = []

        for thread_name, _ in self.threads.items():
            active_threads.append(thread_name)

        return active_threads

    def get_num_active(self) -> int:
        """Get the number of active threads for this processor.

        Returns:
            the number of threads.
        """
        return len(self.threads)

    def is_active_thread(self, thread_name: str) -> bool:
        """Get if the thread with the specified name is active.

        Args:
            thread_name: the name of the thread to query.

        Returns:
            whether the thread is handled by the processor.
        """
        return thread_name in self.threads

    def start(self, thread: ThreadBase) -> None:
        """Start the specified thread.

        The processor takes ownership of the thread and will clean it up
        after it is done running.
        This function returns a KeyError when a thread with the same name
        is already being handled by the processor.

        Args:
            thread: the thread to start.
        """
        thread_name = thread.get_name()
        if self.is_active_thread(thread_name):
            raise KeyError('Thread already active with name:', thread_name)

        self.threads[thread_name] = thread

        thread.start(self.remove)

    def stop(self, thread_name: str) -> None:
        """Stop the thread with the specified name.

        The processor requests the thread to stop running.
        This function returns a KeyError when a thread with the specified name
        is not being handled by the processor.

        Args:
            thread_name: the name of the thread to stop.
        """
        if not self.is_active_thread(thread_name):
            raise KeyError('Thread not active with name:', thread_name)

        # request the thread to stop, it will remove itself after it is done running
        self.threads[thread_name].stop()

    def remove(self, thread):
        """Remove the thread from the processor."""
        del self.threads[thread.get_name()]
