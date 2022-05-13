"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable, Dict, List

from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_FAILURE_ERROR
from ...core.factories import GroupFactory
from ...data.data_transition import DataTransition
from .model_config import ModelConfig


def run_model_pipelines(
        output_dir: str,
        data_transition: DataTransition,
        model_factory: GroupFactory,
        models_config: Dict[str, List[ModelConfig]],
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool],
        **kwargs) -> List[str]:
    """Run several model pipelines for the specified model configurations.

    Args:
        output_dir: the path of the directory to store the output.
        data_transition: data input.
        model_factory: the model factory with available algorithms.
        models_config: containing list of ModelConfig's keyed by API name.
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

    for api_name, models in models_config.items():
        api_factory = model_factory.get_factory(api_name)
        if api_factory is None:
            event_dispatcher.dispatch(
                ON_FAILURE_ERROR,
                msg='Failure: to get algorithm API factory: ' + api_name
            )
            continue

        try:
            model_pipeline = api_factory.create_pipeline(api_factory, event_dispatcher)
            dirs = model_pipeline.run(
                output_dir,
                data_transition,
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
