"""This module contains functionality to run the data pipeline.

Classes:

    DataPipelineConfig: configuration class to run the data pipelines.

Functions:

    run_data_pipelines: run the pipeline using dataset configurations.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Callable, List

from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_FAILURE_ERROR
from ...core.factories import GroupFactory
from ..set.dataset_registry import DataRegistry
from .data_config import DatasetConfig
from .data_pipeline import DataPipeline, DataTransition


@dataclass
class DataPipelineConfig:
    """DataPipeline Configuration."""

    output_dir: str
    data_registry: DataRegistry
    data_factory: GroupFactory
    data_config: List[DatasetConfig]


def run_data_pipelines(
        pipeline_config: DataPipelineConfig,
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool]) -> List[DataTransition]:
    """Run a Data Pipeline several times according to the specified data pipeline configuration.

    Args:
        pipeline_config: the configuration on how to run the data pipelines.
        event_dispatcher: used to dispatch data/IO events when running the pipeline.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Returns:
        a list of DataTransition's.
    """
    data_result = []

    data_pipeline = DataPipeline(pipeline_config.data_factory, event_dispatcher)
    for _, data_config in enumerate(pipeline_config.data_config):
        dataset = pipeline_config.data_registry.get_set(data_config.name)
        if dataset is None:
            event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to get dataset ' + data_config.name + ' from registry'
            )
            continue

        try:
            data_transition = data_pipeline.run(
                pipeline_config.output_dir,
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
