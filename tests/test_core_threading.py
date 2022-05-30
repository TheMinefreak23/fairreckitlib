"""This module tests the threading base/processor functionality.

Classes:

    DummyThread: used for testing assertions in the threading base/processor functionality.

Functions:

    test_thread_base: test the base thread interface through the dummy thread implementation.
    test_thread_processor: test the thread processor functionality using the dummy thread.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

import pytest

from src.fairreckitlib.core.threading.thread_base import ThreadBase
from src.fairreckitlib.core.threading.thread_processor import ThreadProcessor

DUMMY_THREAD_NAME = 'dummy_thread'


class DummyThread(ThreadBase):
    """Dummy thread that runs indefinitely after it is started.

    Added variables are used to assert the correct initialization and termination
    of the ThreadBase functionality.
    """

    def __init__(self, **kwargs):
        """Construct the dummy thread."""
        ThreadBase.__init__(self, DUMMY_THREAD_NAME, True, **kwargs)
        self.initialized = False
        self.terminated = False

    def on_initialize(self) -> None:
        """Initialize the dummy thread."""
        ThreadBase.on_initialize(self)
        assert self.is_running(), 'expected thread to be running before initializing'
        self.initialized = True
        print(self.get_name() + ' initialized')

    def on_run(self, **kwargs) -> None:
        """Run the dummy thread."""
        assert self.initialized, 'expected thread to be initialized before running is invoked'

        # run indefinitely until stop is invoked
        while self.is_running():
            pass

    def on_terminate(self) -> None:
        """Terminate the dummy thread."""
        self.terminated = True
        print(self.get_name() + ' terminated')
        ThreadBase.on_terminate(self)


def test_thread_base() -> None:
    """Test the base thread interface."""
    dummy_thread = DummyThread()

    assert not dummy_thread.is_running(), \
        'did not expect a thread to be running after construction'
    assert dummy_thread.get_name() == DUMMY_THREAD_NAME, \
        'expected the thread name to be the same after construction'

    def on_dummy_thread_terminate(thread: DummyThread) -> None:
        """Call back after the dummy thread is done terminating."""
        assert thread.terminated, 'expected thread to be terminated after running is finished'

    assert dummy_thread.start(on_dummy_thread_terminate), \
        'expected thread to have started'
    assert not dummy_thread.start(on_dummy_thread_terminate), \
        'did not expect thread to start gain while it is already running'

    # simulate active thread for a short amount of time
    time.sleep(1)

    # stop infinite loop
    dummy_thread.stop()


def test_thread_processor() -> None:
    """Test all functionality of the thread processor class."""
    dummy_thread = DummyThread()
    thread_processor = ThreadProcessor()

    def assert_no_active_threads() -> None:
        """Assert the thread processor to have no active threads."""
        assert len(thread_processor.get_active_threads()) == 0, \
            'did not expect any active thread names in the processor'
        assert thread_processor.get_num_active() == 0, \
            'did not expect any active threads in the processor'
        assert not thread_processor.is_active_thread(dummy_thread.get_name()), \
            'did not expect the dummy thread to be active in the processor'

    assert_no_active_threads()
    # test failure on stopping a thread that is not handled by the processor
    pytest.raises(KeyError, thread_processor.stop, dummy_thread.get_name())

    thread_processor.start(dummy_thread)
    # test failure on starting a thread for the second time by the processor
    pytest.raises(KeyError, thread_processor.start, dummy_thread)

    assert thread_processor.get_num_active() == 1, \
        'expected one active thread in the processor'
    assert thread_processor.get_active_threads() == [dummy_thread.get_name()], \
        'expected the dummy thread to be active in the processor'
    assert dummy_thread.is_running(), \
        'expected the dummy thread to be started by the processor'

    # simulate active thread for a short amount of time
    time.sleep(1)

    thread_processor.stop(dummy_thread.get_name())

    # give the processor some time to terminate and dispose of the dummy thread
    time.sleep(1)

    assert_no_active_threads()
