""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import json
import os
from dataclasses import dataclass
from typing import List, Callable

from .evaluation_config import MetricConfig
from ..metrics.common import metric_category_dict
from ...core.event_dispatcher import EventDispatcher
from ...core.event_io import ON_MAKE_DIR
from ...core.factories import GroupFactory
from ...data.data_transition import DataTransition


@dataclass
class EvaluationPipelineConfig:
    """Evaluation Pipeline Configuration."""
    model_dirs: str
    data_transition: DataTransition
    evaluation_factory: GroupFactory
    evaluation: List[MetricConfig]

def run_evaluation_pipelines(
        pipeline_config: EvaluationPipelineConfig,
        event_dispatcher: EventDispatcher,
        is_running: Callable[[], bool],
        **kwargs) -> List[str]:
    """Run several evaluation pipelines according to the specified evaluation pipeline configuration.

    Args:
        pipeline_config: the configuration on how to run the evaluation pipelines.
        event_dispatcher: used to dispatch model/IO events when running the model pipelines.
        is_running: function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Keyword Args:
        num_threads(int): the max number of threads a model can use.
        num_items(int): the number of item recommendations to produce, only
            needed when running recommender pipelines.
    """
    #print('model_dirs', pipeline_config.model_dirs)

    for model_dir in pipeline_config.model_dirs:
        #print('model_dir', model_dir)

        #print('evaluation', pipeline_config.evaluation)
        #api_name = preferred_api_dict[evaluation.name]
        #api_factory = pipeline_config.evaluation_factory.get_factory(api_name)
        #pipeline = api_factory.create_pipeline(api_factory, event_dispatcher)

        recs_path = model_dir + '/ratings.tsv'

        # Create evaluations file
        out_path = os.path.dirname(recs_path) + "/evaluations.json"
        with open(out_path, mode='w', encoding='utf-8') as out_file:
            json.dump({'evaluations': []}, out_file, indent=4)

        event_dispatcher.dispatch(
            ON_MAKE_DIR,
            dir=out_path
        )

        for category, metrics in metric_category_dict.items():
            #print('==DEV CATEGORY METRICS==', category, metrics)
            # Get category metrics
            metrics_names = [metric.value for metric in metrics]
            metrics = [metric for metric in pipeline_config.evaluation if metric.name in metrics_names]
            category_factory = pipeline_config.evaluation_factory.get_factory(category.value)
            pipeline = category_factory.create_pipeline(category_factory, event_dispatcher)
            if not pipeline:
                raise Exception('Category not found')
            pipeline.run(
                out_path,
                recs_path,
                pipeline_config.data_transition,
                metrics,
                is_running,
                **kwargs)
