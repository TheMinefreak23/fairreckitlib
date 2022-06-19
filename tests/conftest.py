"""This module contains constants and fixtures that are shared among all files in the test suite.

Constants:

    DATASET_DIR: the directory to where the dataset samples are stored.
    TMP_DIR: the temporary directory that is used in the io_tmp_dir fixture.
    NUM_THREADS: the (maximum) number of threads used in the pipeline tests.

Fixtures:

    fixture_io_tmp_dir: create and delete temporary directory for other unit tests.
    fixture_io_event_dispatcher: event dispatcher that prints IO events.
    fixture_parse_event_dispatcher: event dispatcher that prints parse events.
    fixture_data_event_dispatcher: event dispatcher that prints data/IO events.
    fixture_eval_event_dispatcher: event dispatcher that prints evaluation/IO events.
    fixture_experiment_event_dispatcher: event dispatcher that prints experiment/IO events.
    fixture_model_event_dispatcher: event dispatcher that prints model/IO events.

Functions:

    is_always_running: function that always returns True to use in pipelines.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

import pytest

from src.fairreckitlib.core.events.event_args import EventArgs
from src.fairreckitlib.core.events.event_dispatcher import EventDispatcher
from src.fairreckitlib.core.events.event_error import \
    get_error_events, get_error_event_print_switch
from src.fairreckitlib.core.io.io_create import create_dir
from src.fairreckitlib.core.io.io_delete import delete_dir
from src.fairreckitlib.core.io.event_io import get_io_events, get_io_event_print_switch
from src.fairreckitlib.core.parsing.parse_event import ON_PARSE, print_parse_event
from src.fairreckitlib.data.pipeline.data_event import \
    get_data_events, get_data_event_print_switch
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.evaluation.pipeline.evaluation_event import \
    get_eval_events, get_eval_event_print_switch
from src.fairreckitlib.experiment.experiment_event import \
    get_experiment_events, get_experiment_print_switch
from src.fairreckitlib.model.pipeline.model_event import \
    get_model_events, get_model_event_print_switch

DATASET_DIR = os.path.join('tests', 'datasets')
TMP_DIR = os.path.join('tests', 'tmp')

NUM_THREADS = 1


@pytest.fixture(scope='package', name='data_registry')
def fixture_data_registry() -> DataRegistry:
    """Fix data registry with (sample) datasets."""
    return DataRegistry(DATASET_DIR)


@pytest.fixture(scope='function', name='io_tmp_dir')
def fixture_io_tmp_dir(io_event_dispatcher: EventDispatcher) -> str:
    """Fix tmp directory creation and deletion for other test functions."""
    create_dir(TMP_DIR, io_event_dispatcher)
    yield TMP_DIR
    delete_dir(TMP_DIR, io_event_dispatcher)


@pytest.fixture(scope='function', name='io_event_dispatcher')
def fixture_io_event_dispatcher() -> EventDispatcher:
    """Fix IO event dispatcher creation for other test functions."""
    print_event = lambda _, args: get_io_event_print_switch()[args.event_id](args)

    event_dispatcher = EventDispatcher()
    for event_id in get_io_events():
        event_dispatcher.add_listener(event_id, None, (print_event, None))

    return event_dispatcher


@pytest.fixture(scope='function', name='parse_event_dispatcher')
def fixture_parse_event_dispatcher() -> EventDispatcher:
    """Fix parse event dispatcher creation for other test functions."""
    event_dispatcher = EventDispatcher()
    event_dispatcher.add_listener(ON_PARSE, None, (lambda _, args: print_parse_event(args), None))
    return event_dispatcher


@pytest.fixture(scope='function', name='data_event_dispatcher')
def fixture_data_event_dispatcher() -> EventDispatcher:
    """Fix data event dispatcher creation for other test functions."""
    def print_data_pipeline_event(_, event_args: EventArgs, **kwargs) -> None:
        """Print the data event using print switches."""
        print_switch = get_data_event_print_switch(kwargs.get('elapsed_time'))
        print_switch.update(get_error_event_print_switch())
        print_switch.update(get_io_event_print_switch())
        print_switch[event_args.event_id](event_args)

    events = get_error_events() + get_io_events() + get_data_events()
    event_dispatcher = EventDispatcher()

    for event_id in events:
        event_dispatcher.add_listener(event_id, None, (print_data_pipeline_event, None))

    return event_dispatcher


@pytest.fixture(scope='function', name='eval_event_dispatcher')
def fixture_eval_event_dispatcher() -> EventDispatcher:
    """Fix evaluation event dispatcher creation for other test functions."""
    def print_eval_pipeline_event(_, event_args: EventArgs, **kwargs) -> None:
        """Print the eval event using print switches."""
        print_switch = get_eval_event_print_switch(kwargs.get('elapsed_time'))
        print_switch.update(get_error_event_print_switch())
        print_switch.update(get_io_event_print_switch())
        print_switch[event_args.event_id](event_args)

    events = get_error_events() + get_io_events() + get_eval_events()
    event_dispatcher = EventDispatcher()

    for event_id in events:
        event_dispatcher.add_listener(event_id, None, (print_eval_pipeline_event, None))

    return event_dispatcher


@pytest.fixture(scope='function', name='experiment_event_dispatcher')
def fixture_experiment_event_dispatcher() -> EventDispatcher:
    """Fix experiment event dispatcher creation for other test functions."""
    def print_experiment_pipeline_event(_, event_args: EventArgs, **kwargs) -> None:
        """Print the experiment event using print switches."""
        print_switch = get_experiment_print_switch(kwargs.get('elapsed_time'))
        print_switch[event_args.event_id](event_args)

    event_dispatcher = EventDispatcher()

    for event_id in get_experiment_events():
        event_dispatcher.add_listener(event_id, None, (print_experiment_pipeline_event, None))

    return event_dispatcher


@pytest.fixture(scope='function', name='model_event_dispatcher')
def fixture_model_event_dispatcher() -> EventDispatcher:
    """Fix model event dispatcher creation for other test functions."""
    def print_model_pipeline_event(_, event_args: EventArgs, **kwargs) -> None:
        """Print the model event using print switches."""
        print_switch = get_model_event_print_switch(kwargs.get('elapsed_time'))
        print_switch.update(get_error_event_print_switch())
        print_switch.update(get_io_event_print_switch())
        print_switch[event_args.event_id](event_args)

    events = get_error_events() + get_io_events() + get_model_events()
    event_dispatcher = EventDispatcher()

    for event_id in events:
        event_dispatcher.add_listener(event_id, None, (print_model_pipeline_event, None))

    return event_dispatcher


def is_always_running() -> bool:
    """Return True."""
    return True
