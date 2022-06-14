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
from ...data.data_transition import DataTransition
from .evaluation_config import MetricConfig
from .evaluation_pipeline import EvaluationPipeline, EvaluationSetPaths


@dataclass
class EvaluationPipelineConfig:
    """Evaluation Pipeline Configuration.

    output_dir: the directory to store the output.
    data_transition: data input.
    data_filter_factory: the factory with available filters for all dataset-matrix pairs.
    metric_category_factory: the factory with available metric category factories.
    evaluation: list of metric configurations to compute.
    """

    model_dirs: List[str]
    data_transition: DataTransition
    data_filter_factory: GroupFactory
    metric_category_factory: GroupFactory
    evaluation: List[MetricConfig]


def run_evaluation_pipelines(
        pipeline_config: EvaluationPipelineConfig,
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool]) -> None:
    """Run several evaluation pipelines according to the specified eval pipeline configuration.

    Args:
        pipeline_config: the configuration on how to run the evaluation pipelines.
        event_dispatcher: used to dispatch eval/IO events when running the evaluation pipelines.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.
    """
    data_transition = pipeline_config.data_transition

    eval_pipeline = EvaluationPipeline(
        data_transition.dataset,
        pipeline_config.data_filter_factory,
        pipeline_config.metric_category_factory,
        event_dispatcher
    )

    # remove any metrics that have a subgroup that is not related to the data transition
    metric_config_list = [metric for metric in pipeline_config.evaluation
                          if metric.subgroup is None or \
                          metric.subgroup.dataset == data_transition.dataset.get_name() and \
                          metric.subgroup.matrix == data_transition.matrix_name]

    for model_dir in pipeline_config.model_dirs:
        rating_set_path = os.path.join(model_dir, MODEL_RATINGS_FILE)
        output_path = os.path.join(model_dir, 'evaluations.json')

        try:
            eval_pipeline.run(
                output_path,
                EvaluationSetPaths(
                    data_transition.train_set_path,
                    data_transition.test_set_path,
                    rating_set_path
                ),
                metric_config_list,
                is_running
            )
        except FileNotFoundError:
            continue
