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

from ...core.config.config_factories import GroupFactory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.events.event_error import ON_FAILURE_ERROR, ErrorEventArgs
from ..set.dataset_registry import DataRegistry
from .data_config import DataMatrixConfig
from .data_pipeline import DataPipeline, DataTransition


@dataclass
class DataPipelineConfig:
    """Data Pipeline Configuration.

    output_dir: the directory to store the output.
    data_registry: the registry with available datasets.
    data_factory: the factory with available data modifier factories.
    data_config_list: the dataset matrix configurations to compute.
    """

    output_dir: str
    data_registry: DataRegistry
    data_factory: GroupFactory
    data_config_list: List[DataMatrixConfig]


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
    for data_config in pipeline_config.data_config_list:
        dataset = pipeline_config.data_registry.get_set(data_config.dataset)
        if dataset is None:
            event_dispatcher.dispatch(ErrorEventArgs(
                ON_FAILURE_ERROR,
                'Failure: to get dataset ' + data_config.dataset + ' from registry'
            ))
            continue

        if data_config.matrix not in dataset.get_available_matrices():
            event_dispatcher.dispatch(ErrorEventArgs(
                ON_FAILURE_ERROR,
                'Failure: to get matrix ' + data_config.matrix + ' from ' + data_config.dataset
            ))
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
