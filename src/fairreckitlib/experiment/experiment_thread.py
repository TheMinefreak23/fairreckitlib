"""This module contains functionality to execute the experiment pipelines on a thread.

Classes:

    ThreadExperiment: class that runs the experiment pipelines on a (closable) thread.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Callable, Dict

from ..core.threading.thread_base import ThreadBase
from .experiment_event import get_experiment_events
from .experiment_event import ON_BEGIN_EXPERIMENT_THREAD, ON_END_EXPERIMENT_THREAD
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
        for (event_id, on_event) in get_experiment_events():
            func_on_event = (on_event, events.get(event_id))
            self.event_dispatcher.add_listener(event_id, self, func_on_event)

    def on_run(self, **kwargs):
        """Run the experiment pipeline.

        Keyword Args:
            pipeline_config(ExperimentPipelineConfig): configuration of the experiment pipeline.
        """
        pipeline_config = kwargs['pipeline_config']

        self.event_dispatcher.dispatch(
            ON_BEGIN_EXPERIMENT_THREAD,
            num_runs=pipeline_config.num_runs,
            experiment_name=pipeline_config.experiment_config.name
        )

        start = time.time()

        run_experiment_pipelines(pipeline_config, self.event_dispatcher, self.is_running)

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_EXPERIMENT_THREAD,
            num_runs=pipeline_config.num_runs,
            aborted=self.is_running(),
            experiment_name=pipeline_config.experiment_config.name,
            elapsed_time=end-start,
        )
