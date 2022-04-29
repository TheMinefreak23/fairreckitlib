"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


class ThreadProcessor:
    """Processor for multiple threads (derived from ThreadBase class).

    Keeps track of all threads that are started by this processor.
    Additionally, these started threads can be stopped as well.
    """
    def __init__(self):
        self.__threads = {}

    def get_num_active(self):
        """Gets the number of active threads for this processor.

        Returns:
            (int): the number of threads.
        """
        return len(self.__threads)

    def is_active_thread(self, thread_name):
        """Returns whether the thread with the specified name is active.

        Returns:
            (bool): whether the thread is handled by the processor.
        """
        return thread_name in self.__threads

    def start(self, thread):
        """Starts the specified thread.

        The processor takes ownership of the thread and will clean it up
        after it is done running.
        This function returns a KeyError when a thread with the same name
        is already being handled by the processor.

        Args:
            thread(ThreadBase): the thread to start.
        """
        thread_name = thread.get_name()
        if self.is_active_thread(thread_name):
            raise KeyError('Thread already active with name:', thread_name)

        self.__threads[thread_name] = thread

        thread.start(self.__remove)

    def stop(self, thread_name):
        """Stops the thread with the specified name.

        The processor requests the thread to stop running.
        This function returns a KeyError when a thread with the specified name
        is not being handled by the processor.

        Args:
            thread_name(str): the name of the thread to stop.
        """
        if not self.is_active_thread(thread_name):
            raise KeyError('Thread not active with name:', thread_name)

        # request the thread to stop, it will remove itself after it is done running
        self.__threads[thread_name].stop()

    def __remove(self, thread):
        """Removes the thread from the processor."""
        del self.__threads[thread.get_name()]
