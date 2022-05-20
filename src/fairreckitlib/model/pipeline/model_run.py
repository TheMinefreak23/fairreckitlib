"""This module contains functionality that wraps running the model pipeline multiple times.

Classes:

    ModelPipelineConfig: configuration class to run the model pipelines.

Functions:

    run_model_pipelines: run (multiple) pipelines for specified model configurations.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_FAILURE_ERROR
from ...core.factories import GroupFactory
from ...data.data_transition import DataTransition
from .model_config import ModelConfig


@dataclass
class ModelPipelineConfig:
    """ModelPipeline Configuration."""

    output_dir: str
    data_transition: DataTransition
    model_factory: GroupFactory
    models: Dict[str, List[ModelConfig]]


def run_model_pipelines(
        pipeline_config: ModelPipelineConfig,
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool],
        **kwargs) -> List[str]:
    """Run several model pipelines according to the specified model pipeline configuration.

    Args:
        pipeline_config: the configuration on how to run the model pipelines.
        event_dispatcher: used to dispatch model/IO events when running the model pipelines.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Keyword Args:
        num_threads(int): the max number of threads a model can use.
        num_items(int): the number of item recommendations to produce, only
            needed when running recommender pipelines.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        list of directories where the computed model ratings are stored.
    """
    model_dirs = []

    for api_name, models in pipeline_config.models.items():
        api_factory = pipeline_config.model_factory.get_factory(api_name)
        if api_factory is None:
            event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to get algorithm API factory: ' + api_name
            )
            continue

        try:
            model_pipeline = api_factory.create_pipeline(api_factory, event_dispatcher)
            dirs = model_pipeline.run(
                pipeline_config.output_dir,
                pipeline_config.data_transition,
                models,
                is_running,
                **kwargs
            )
        except FileNotFoundError:
            continue

        model_dirs += dirs

        if not is_running():
            return model_dirs

    return model_dirs
