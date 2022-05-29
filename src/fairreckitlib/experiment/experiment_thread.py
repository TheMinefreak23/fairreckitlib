"""This module contains functionality to execute the experiment pipelines on a thread.

Classes:

    ThreadExperiment: class that runs the experiment pipelines on a (closable) thread.

Functions:

    handle_experiment_event: handles incoming experiment events.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Callable, Dict

from ..core.events.event_dispatcher import EventArgs
from ..core.threading.thread_base import ThreadBase
from .experiment_event import get_experiment_events, get_experiment_print_switch, \
    ExperimentThreadEventArgs, ON_BEGIN_EXPERIMENT_THREAD, ON_END_EXPERIMENT_THREAD
from .experiment_run import run_experiment_pipelines


class ThreadExperiment(ThreadBase):
    """Thread that runs the same experiment one or more times."""

    def __init__(
            self,
            name: str,
            events: Dict[Any, Callable[[Any], None]]=None,
            verbose: bool=False,
            **kwargs):
        """Construct the ExperimentThread.

        Args:
            name the name of the thread.
            events: events to dispatch for this thread.
            verbose: whether the thread should give verbose output.

        Keyword Args:
            pipeline_config(ExperimentPipelineConfig): configuration of the experiment pipeline.
        """
        ThreadBase.__init__(self, name, verbose, **kwargs)
        # no external events specified
        if events is None:
            events = {}

        # Add external events.
        for event_id in get_experiment_events():
            func_on_event = (handle_experiment_event, events.get(event_id))
            self.event_dispatcher.add_listener(event_id, self, func_on_event)

    def on_run(self, **kwargs):
        """Run the experiment pipeline.

        Keyword Args:
            pipeline_config(ExperimentPipelineConfig): configuration of the experiment pipeline.
        """
        pipeline_config = kwargs['pipeline_config']

        self.event_dispatcher.dispatch(ExperimentThreadEventArgs(
            ON_BEGIN_EXPERIMENT_THREAD,
            pipeline_config.experiment_config.name,
            pipeline_config.num_runs,
            self.is_running()
        ))

        start = time.time()

        run_experiment_pipelines(pipeline_config, self.event_dispatcher, self.is_running)

        end = time.time()

        self.event_dispatcher.dispatch(ExperimentThreadEventArgs(
            ON_END_EXPERIMENT_THREAD,
            pipeline_config.experiment_config.name,
            pipeline_config.num_runs,
            self.is_running()
        ), elapsed_time=end - start)


def handle_experiment_event(
        experiment_thread: ThreadExperiment,
        event_args: EventArgs,
        **kwargs) -> None:
    """Handle incoming experiment events.

    It is assumed that the event finished when the elapsed_time keyword argument is available.

    Args:
        experiment_thread: the listening experiment thread.
        event_args: the event arguments to handle.

    Keyword Args:
        elapsed_time(float): time that has passed since the event started, expressed in seconds.
    """
    if experiment_thread.verbose:
        event_switch = get_experiment_print_switch(kwargs.get('elapsed_time'))
        event_switch[event_args.event_id](event_args)
