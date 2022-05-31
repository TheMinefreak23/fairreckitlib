"""This module contains all event ids, event args and a print switch for the experiment pipeline.

Constants:

    ON_BEGIN_EXPERIMENT_PIPELINE: id of the event that is used when the experiment pipeline starts.
    ON_END_EXPERIMENT_PIPELINE: id of the event that is used when the experiment pipeline ends.
    ON_BEGIN_THREAD_EXPERIMENT: id of the event that is used when the experiment thread starts.
    ON_END_THREAD_EXPERIMENT: id of the event that is used when the experiment thread ends.

Classes:

    ExperimentEventArgs: event args related to an experiment.
    ExperimentThreadEventArgs: event args related to an experiment thread.

Functions:

    get_experiment_events: list of experiment pipeline event IDs.
    get_experiment_event_print_switch: switch to print experiment pipeline event arguments by ID.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Dict, Callable, List

from ..core.events.event_dispatcher import EventArgs
from ..core.events.event_error import get_error_events, get_error_event_print_switch
from ..core.io.event_io import get_io_events, get_io_event_print_switch
from ..data.pipeline.data_event import get_data_events, get_data_event_print_switch
from ..evaluation.pipeline.evaluation_event import get_eval_events, get_eval_event_print_switch
from ..model.pipeline.model_event import get_model_events, get_model_event_print_switch

ON_BEGIN_EXPERIMENT_PIPELINE = 'Experiment.on_begin_pipeline'
ON_END_EXPERIMENT_PIPELINE = 'Experiment.on_end_pipeline'
ON_BEGIN_EXPERIMENT_THREAD = 'Experiment.on_begin_thread'
ON_END_EXPERIMENT_THREAD = 'Experiment.on_end_thread'


@dataclass
class ExperimentEventArgs(EventArgs):
    """Experiment Event Arguments.

    event_id: the unique ID that classifies the experiment event.
    experiment_name: the name of the experiment.
    """

    experiment_name: str


@dataclass
class ExperimentThreadEventArgs(ExperimentEventArgs):
    """Experiment Thread Event Arguments.

    event_id: the unique ID that classifies the experiment event.
    experiment_name: the name of the experiment.
    num_runs: the amount of times the experiment will run.
    is_running: whether the experiment thread is running or aborted.
    """

    num_runs: int
    is_running: bool=True


def get_experiment_events() -> List[str]:
    """Get a list of experiment pipeline event IDs.

    Returns:
        a list of unique experiment pipeline event IDs.
    """
    events = []
    events += get_error_events()
    events += get_io_events()
    events += get_data_events()
    events += get_model_events()
    events += get_eval_events()
    return events


def get_experiment_print_switch(elapsed_time: float=None) -> Dict[str, Callable[[EventArgs], None]]:
    """Get a switch that prints experiment pipeline event IDs.

    Returns:
        the print experiment pipeline event switch.
    """
    event_switch = {
        # experiment thread events
        ON_BEGIN_EXPERIMENT_THREAD: lambda args:
            print('Starting', args.num_runs, 'experiment(s) with name', args.experiment_name),
        ON_END_EXPERIMENT_THREAD: lambda args:
            print('Finished', args.num_runs, 'experiment(s) with name', args.experiment_name,
                  f'in {elapsed_time:1.4f}s'),
        # experiment pipeline events
        ON_BEGIN_EXPERIMENT_PIPELINE: lambda args:
            print('Starting Experiment:', args.experiment_name),
        ON_END_EXPERIMENT_PIPELINE: lambda args:
            print('Finished Experiment:', args.experiment_name,
                  f'in {elapsed_time:1.4f}s')
    }

    # merge error/IO/pipeline event switches
    event_switch.update(get_error_event_print_switch())
    event_switch.update(get_io_event_print_switch())
    event_switch.update(get_data_event_print_switch(elapsed_time))
    event_switch.update(get_model_event_print_switch(elapsed_time))
    event_switch.update(get_eval_event_print_switch(elapsed_time))

    return event_switch
