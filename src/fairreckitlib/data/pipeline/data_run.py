"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import List, Dict, Any
from ...core.event_dispatcher import EventDispatcher
from ...core.factories import BaseFactory
from ..set.dataset_registry import DataRegistry
from .data_pipeline import DataPipeline, DataTransition



def run_data_pipeline(output_dir: str, data_registry: DataRegistry, split_factory: BaseFactory,
                      datasets_config: List[Dict[str, Any]], event_dispatcher: EventDispatcher,
                      is_running: function) -> List[DataTransition]:
    """Run the Data Pipeline for multiple dataset configurations.

    Args:
        output_dir: the path of the directory to store the output.
        data_registry: the registry of available datasets.
        split_factory: factory of available splitters.
        datasets_config: list of DatasetConfig objects.
        event_dispatcher: used to dispatch data/IO events
            when running the pipeline.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Returns:
        data_result: list of DataTransitions.
    """
    data_result = []

    data_pipeline = DataPipeline(split_factory, event_dispatcher)
    for _, data_config in enumerate(datasets_config):
        dataset = data_registry.get_set(data_config.name)

        data_result.append(data_pipeline.run(
            output_dir,
            dataset,
            data_config,
            is_running
        ))
        if not is_running():
            return None

    return data_result
