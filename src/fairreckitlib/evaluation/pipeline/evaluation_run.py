""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from ..metrics.common import RecType
from ..metrics.common import Test
from .evaluation_pipeline import EvaluationPipeline


def run_evaluation_pipelines(model_dirs, data_transition, metric_factory,
                             eval_config, event_dispatcher, is_running, **kwargs):
    """Runs several ModelPipeline's for the specified model configurations.

    Args:
        model_dirs(array like): list of directories where the computed model
            ratings are stored.
        data_transition(DataTransition): data input.
        metric_factory(GroupFactory): the metric factory with available metrics.
        eval_config(array like): containing list of MetricConfig's.
        event_dispatcher(EventDispatcher): used to dispatch evaluation/IO events
            when running the evaluation pipelines.
        is_running(func -> bool): function that returns whether the pipelines
            are still running. Stops early when False is returned.

    Keyword Args:
        num_threads(int): the max number of threads the evaluation can use.
        num_items(int): the number of item recommendations to produce, only
            needed when running recommender pipelines.
    """
    print('model_dirs',model_dirs)

    for model_dir in model_dirs:
        print('model_dir',model_dir)

        for evaluations in eval_config:
            api_factory = pipeline_config.model_factory.get_factory(api_name)
            data_paths = (data_transition, model_dir+'/ratings.tsv')
            pipeline =
            pipeline.run(data_paths, eval_config, event_dispatcher)
