""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
from dataclasses import dataclass
from typing import List

from .evaluation_config import MetricConfig
from ..metrics.common import RecType, Metric, metric_category_dict
from ..metrics.common import Test
from .evaluation_pipeline import EvaluationPipeline
from ...core.apis import LENSKIT_API, REXMEX_API
from ...core.factories import GroupFactory
from ...data.data_transition import DataTransition


@dataclass
class EvaluationPipelineConfig:
    """Evaluation Pipeline Configuration."""
    model_dirs: str
    data_transition: DataTransition
    evaluation_factory: GroupFactory
    evaluation: List[MetricConfig]


"""
preferred_api_dict = {
    LENSKIT_API: [Metric.NDCG,
                  Metric.PRECISION,
                  Metric.RECALL,
                  Metric.MRR,
                  Metric.RMSE,
                  Metric.MAE
                  ],
    REXMEX_API: [Metric.ITEM_COVERAGE,
                 Metric.USER_COVERAGE,
                 Metric.INTRA_LIST_SIMILARITY,
                 Metric.NOVELTY]
}"""

preferred_api_dict = {
    Metric.NDCG.value: LENSKIT_API,
    Metric.PRECISION.value: LENSKIT_API,
    Metric.RECALL.value: LENSKIT_API,
    Metric.MRR.value: LENSKIT_API,
    Metric.RMSE.value: LENSKIT_API,
    Metric.MAE.value: LENSKIT_API,
    Metric.ITEM_COVERAGE.value: REXMEX_API,
    Metric.USER_COVERAGE.value: REXMEX_API,
    Metric.INTRA_LIST_SIMILARITY.value: REXMEX_API,
    Metric.NOVELTY.value: REXMEX_API
}


def run_evaluation_pipelines(
        pipeline_config: EvaluationPipelineConfig,
        event_dispatcher,
        is_running,
        **kwargs):
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
    print('model_dirs', pipeline_config.model_dirs)

    for model_dir in pipeline_config.model_dirs:
        print('model_dir', model_dir)

        for evaluation in pipeline_config.evaluation:
            print(evaluation)
            #api_name = preferred_api_dict[evaluation.name]
            #api_factory = pipeline_config.evaluation_factory.get_factory(api_name)
            #pipeline = api_factory.create_pipeline(api_factory, event_dispatcher)
            pipeline = None
            # Find category for the metric
            for category, metrics in metric_category_dict.items():
                if evaluation.name in [metric.value for metric in metrics]:
                    print(category, metrics)
                    category_factory = pipeline_config.evaluation_factory.get_factory(category.value)
                    print('DEV line 95 evaluation_run', category_factory)
                    pipeline = category_factory.create_pipeline(category_factory, event_dispatcher)
            pipeline.run(
                model_dir + '/ratings.tsv',
                pipeline_config.data_transition,
                pipeline_config.evaluation,
                is_running,
                **kwargs)
        # pipeline = pipeline_config.evaluation_factory.get_factory()
