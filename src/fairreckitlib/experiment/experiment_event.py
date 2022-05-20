"""This module contains all event ids and callback functions used in the experiment pipeline.

Constants:

    ON_BEGIN_EXPERIMENT_PIPELINE: id of the event that is used when the experiment pipeline starts.
    ON_END_EXPERIMENT_PIPELINE: id of the event that is used when the experiment pipeline ends.
    ON_BEGIN_THREAD_EXPERIMENT: id of the event that is used when the experiment thread starts.
    ON_END_THREAD_EXPERIMENT: id of the event that is used when the experiment thread ends.

Functions:

    get_experiment_events: get experiment pipeline events.
    on_begin_experiment_pipeline: call when the experiment pipeline starts.
    on_end_experiment_pipeline: call when the experiment pipeline ends.
    on_begin_experiment_thread: call when the experiment thread starts.
    on_end_experiment_thread: call when the experiment thread ends.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, List, Tuple

from ..core.event_error import get_error_events
from ..core.event_io import get_io_events
from ..data.pipeline.data_event import get_data_events
from ..evaluation.pipeline.evaluation_event import get_evaluation_events
from ..model.pipeline.model_event import get_model_events


ON_BEGIN_EXPERIMENT_PIPELINE = 'Experiment.on_begin_pipeline'
ON_END_EXPERIMENT_PIPELINE = 'Experiment.on_end_pipeline'
ON_BEGIN_EXPERIMENT_THREAD = 'Experiment.on_begin_thread'
ON_END_EXPERIMENT_THREAD = 'Experiment.on_end_thread'


def get_experiment_events() -> List[Tuple[str, Callable[[Any], None]]]:
    """Get all experiment pipeline events.

    The callback functions are specified below and serve as a default
    implementation for the RecommenderSystem class including the keyword arguments
    that are passed down by the data pipeline.

    Returns:
        a list of pairs in the format (event_id, func_on_event)
    """
    events = [
        (ON_BEGIN_EXPERIMENT_PIPELINE, on_begin_experiment_pipeline),
        (ON_END_EXPERIMENT_PIPELINE, on_end_experiment_pipeline),
        (ON_BEGIN_EXPERIMENT_THREAD, on_begin_experiment_thread),
        (ON_END_EXPERIMENT_THREAD, on_end_experiment_thread)
    ]

    events += get_error_events()
    events += get_io_events()
    events += get_data_events()
    events += get_model_events()
    events += get_evaluation_events()
    return events


def on_begin_experiment_pipeline(event_listener: Any, **kwargs) -> None:
    """Call back when the experiment pipeline starts.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        experiment_name(str): name of the experiment.
    """
    if event_listener.verbose:
        print('Starting Experiment:', kwargs['experiment_name'])


def on_end_experiment_pipeline(event_listener: Any, **kwargs) -> None:
    """Call back when an experiment finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        experiment_name(str): name of the experiment.
        elapsed_time(float): the time that has passed since the model
            computation started, expressed in seconds.
    """
    if event_listener.verbose:
        elapsed_time = kwargs['elapsed_time']
        experiment_name = kwargs['experiment_name']
        print('Finished Experiment:', experiment_name, f'in {elapsed_time:1.4f}s')


def on_begin_experiment_thread(event_listener: Any, **kwargs) -> None:
    """Call back when an experiment thread started.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_runs(int): number of experiments to run.
        experiment_name(str): name of the experiment (run).
    """
    if event_listener.verbose:
        print('Starting', kwargs['num_runs'], 'experiment(s) with name', kwargs['experiment_name'])


def on_end_experiment_thread(event_listener: Any, **kwargs) -> None:
    """Call back when an experiment thread finished.

    Args:
        event_listener: the listener that is registered
            in the event dispatcher with this callback.

    Keyword Args:
        num_runs(int): number of experiments that ran.
        aborted(bool): whether the experiment thread was aborted.
        experiment_name(str): name of the experiment (run).
        elapsed_time(float): the time that has passed since the model
            computation started, expressed in seconds.
    """
    if event_listener.verbose:
        print('Finished', kwargs['num_runs'], 'experiment(s) with name', kwargs['experiment_name'])
