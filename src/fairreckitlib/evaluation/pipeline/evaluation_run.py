"""This module contains functionality that wraps running the evaluation pipeline multiple times.

Classes:

    EvaluationPipelineConfig: configuration class to run the evaluation pipelines.

Functions:

    run_evaluation_pipelines: run (multiple) pipelines for specified metric configurations.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
import os
from typing import List, Callable

from ...core.config.config_factories import GroupFactory
from ...core.core_constants import MODEL_RATINGS_FILE
from ...core.events.event_dispatcher import EventDispatcher
from ...core.io.io_create import create_json
from ...data.data_transition import DataTransition
from ..metrics.common import metric_category_dict
from .evaluation_config import MetricConfig


@dataclass
class EvaluationPipelineConfig:
    """Evaluation Pipeline Configuration.

    output_dir: the directory to store the output.
    data_transition: data input.
    evaluation_factory: the factory with available group metric factories.
    evaluation: list of metric configurations to compute.
    """

    model_dirs: List[str]
    data_transition: DataTransition
    evaluation_factory: GroupFactory
    evaluation: List[MetricConfig]


def run_evaluation_pipelines(
        pipeline_config: EvaluationPipelineConfig,
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool],
        **kwargs) -> None:
    """Run several evaluation pipelines according to the specified eval pipeline configuration.

    Args:
        pipeline_config: the configuration on how to run the evaluation pipelines.
        event_dispatcher: used to dispatch eval/IO events when running the evaluation pipelines.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Keyword Args:
        num_threads(int): the max number of threads an evaluation can use.
        num_items(int): the number of item recommendations to produce, only
            needed when running recommender pipelines.
    """
    for model_dir in pipeline_config.model_dirs:
        #print('evaluation', pipeline_config.evaluation)
        #api_name = preferred_api_dict[evaluation.name]
        #api_factory = pipeline_config.evaluation_factory.get_factory(api_name)
        #pipeline = api_factory.create_pipeline(api_factory, event_dispatcher)

        recs_path = os.path.join(model_dir, MODEL_RATINGS_FILE)
        out_path = os.path.join(model_dir, 'evaluations.json')

        # Create evaluations file
        create_json(
            out_path,
            {'evaluations': []},
            event_dispatcher,
            indent=4
        )

        for category, metrics in metric_category_dict.items():
            #print('==DEV CATEGORY METRICS==', category, metrics)
            # Get category metrics
            metrics_names = [metric.value for metric in metrics]
            metrics = [metric for metric in pipeline_config.evaluation
                       if metric.name in metrics_names]
            category_factory = pipeline_config.evaluation_factory.get_factory(category.value)
            pipeline = category_factory.create_pipeline(category_factory, event_dispatcher)
            if not pipeline:
                raise Exception('Category not found')
            if len(metrics) == 0:
                continue

            kwargs['is_running'] = is_running

            pipeline.run(
                out_path,
                recs_path,
                pipeline_config.data_transition,
                metrics,
                **kwargs)
