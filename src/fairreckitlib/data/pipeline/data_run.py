"""This module contains functionality to run the data pipeline.

Functions:

    run_data_pipeline: run the pipeline using dataset configurations.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable, List

from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_FAILURE_ERROR
from ...core.factories import GroupFactory
from ..set.dataset_registry import DataRegistry
from .data_config import DatasetConfig
from .data_pipeline import DataPipeline, DataTransition


def run_data_pipeline(
        output_dir: str,
        data_registry: DataRegistry,
        data_factory: GroupFactory,
        datasets_config: List[DatasetConfig],
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool]) -> List[DataTransition]:
    """Run the Data Pipeline for multiple dataset configurations.

    Args:
        output_dir: the path of the directory to store the output.
        data_registry: the registry with available datasets.
        data_factory: the factory with available data modifier factories.
        datasets_config: list of DatasetConfig objects.
        event_dispatcher: used to dispatch data/IO events when running the pipeline.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Returns:
        a list of DataTransition's .
    """
    data_result = []

    data_pipeline = DataPipeline(data_factory, event_dispatcher)
    for _, data_config in enumerate(datasets_config):
        dataset = data_registry.get_set(data_config.name)
        if dataset is None:
            event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to get dataset ' + data_config.name + ' from registry'
            )
            continue

        try:
            data_transition = data_pipeline.run(
                output_dir,
                dataset,
                data_config,
                is_running
            )
        except (FileNotFoundError, RuntimeError):
            continue

        data_result.append(data_transition)
        if not is_running():
            return data_result

    return data_result
