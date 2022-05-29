"""This module contains all event ids, event args and a print switch for the evaluation pipeline.

Constants:

    ON_BEGIN_EVAL_PIPELINE: id of the event that is used when the evaluation pipeline starts.
    ON_BEGIN_FILTER_RECS: id of the event that is used when recs filtering starts.
    ON_BEGIN_LOAD_RECS_SET: id of the event that is used when a recs set is being loaded.
    ON_BEGIN_LOAD_TEST_SET: id of the event that is used when a test set is being loaded.
    ON_BEGIN_LOAD_TRAIN_SET: id of the event that is used when a train set is being loaded.
    ON_BEGIN_METRIC: id of the event that is used when a metric computation started.
    ON_END_EVAL_PIPELINE: id of the event that is used when the evaluation pipeline ends.
    ON_END_FILTER_RECS: id of the event that is used when recs filtering finishes.
    ON_END_LOAD_RECS_SET: id of the event that is used when a recs set has been loaded.
    ON_END_LOAD_TEST_SET: id of the event that is used when a test set has been loaded.
    ON_END_LOAD_TRAIN_SET: id of the event that is used when a train set has been loaded.
    ON_END_METRIC: id of the event that is used when a metric computation finishes.

Classes:

    EvaluationPipelineEventArgs: event args related to the evaluation pipeline.
    MetricEventArgs: event args related to a metric.

Functions:

    get_eval_events: list of evaluation pipeline event IDs.
    get_eval_event_print_switch: switch to print evaluation pipeline event arguments by ID.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from ...core.events.event_dispatcher import EventArgs
from ...core.events.event_io import print_load_df_event_args
from ...data.filter.filter_event import print_filter_event_args
from .evaluation_config import MetricConfig

ON_BEGIN_LOAD_TEST_SET = 'EvaluationPipeline.on_begin_load_test_set'
ON_END_LOAD_TEST_SET = 'EvaluationPipeline.on_end_load_test_set'
ON_BEGIN_LOAD_TRAIN_SET = 'EvaluationPipeline.on_begin_load_train_set'
ON_END_LOAD_TRAIN_SET = 'EvaluationPipeline.on_end_load_train_set'
ON_BEGIN_LOAD_RECS_SET = 'EvaluationPipeline.on_begin_load_recs_set'
ON_END_LOAD_RECS_SET = 'EvaluationPipeline.on_end_load_recs_set'
ON_BEGIN_EVAL_PIPELINE = 'EvaluationPipeline.on_begin_eval_pipeline'
ON_END_EVAL_PIPELINE = 'EvaluationPipeline.on_end_eval_pipeline'
ON_BEGIN_METRIC = 'EvaluationPipeline.on_begin_metric'
ON_END_METRIC = 'EvaluationPipeline.on_end_metric'
ON_BEGIN_FILTER_RECS = 'EvaluationPipeline.on_begin_filter_recs'
ON_END_FILTER_RECS = 'EvaluationPipeline.on_end_filter_recs'


@dataclass
class EvaluationPipelineEventArgs(EventArgs):
    """Evaluation Pipeline Event Arguments.

    event_id: the unique ID that classifies the evaluation pipeline event.
    metrics_config: list of metric configurations that is used in the evaluation pipeline.
    """

    metrics_config: List[MetricConfig]


@dataclass
class MetricEventArgs(EventArgs):
    """Evaluation Pipeline Event Arguments.

    event_id: the unique ID that classifies the metric event.
    metric_config: the metric configuration that is used.
    """

    metric_config: MetricConfig


def get_eval_events() -> List[str]:
    """Get a list of evaluation pipeline event IDs.

    Returns:
        a list of unique evaluation pipeline event IDs.
    """
    return [
        # DataframeEventArgs
        ON_BEGIN_LOAD_TEST_SET,
        ON_END_LOAD_TEST_SET,
        # DataframeEventArgs
        ON_BEGIN_LOAD_TRAIN_SET,
        ON_END_LOAD_TRAIN_SET,
        # DataframeEventArgs
        ON_BEGIN_LOAD_RECS_SET,
        ON_END_LOAD_RECS_SET,
        # EvaluationPipelineEventArgs
        ON_BEGIN_EVAL_PIPELINE,
        ON_END_EVAL_PIPELINE,
        # MetricEventArgs
        ON_BEGIN_METRIC,
        ON_END_METRIC,
        # FilterDataframeEventArgs
        ON_BEGIN_FILTER_RECS,
        ON_END_FILTER_RECS,
    ]


def get_eval_event_print_switch(elapsed_time: float=None) -> Dict[str, Callable[[EventArgs], None]]:
    """Get a switch that prints evaluation pipeline event IDs.

    Returns:
        the print evaluation pipeline event switch.
    """
    return {
        ON_BEGIN_EVAL_PIPELINE:
            lambda args: print('\nStarting Evaluation Pipeline to process',
                               len(args.metrics_config), 'metric(s)'),
        ON_BEGIN_FILTER_RECS: print_filter_event_args,
        ON_BEGIN_LOAD_RECS_SET: print_load_df_event_args,
        ON_BEGIN_LOAD_TEST_SET: print_load_df_event_args,
        ON_BEGIN_LOAD_TRAIN_SET: print_load_df_event_args,
        ON_BEGIN_METRIC:
            lambda args: print('Starting evaluation with metric', args.metric_config.name),
        ON_END_EVAL_PIPELINE:
            lambda args: print('Finished Evaluation Pipeline on',
                               len(args.metrics_config), 'metrics',
                               f'in {elapsed_time:1.4f}s'),
        ON_END_FILTER_RECS: lambda args: print_filter_event_args(args, elapsed_time),
        ON_END_LOAD_RECS_SET: lambda args: print_load_df_event_args(args, elapsed_time),
        ON_END_LOAD_TEST_SET: lambda args: print_load_df_event_args(args, elapsed_time),
        ON_END_LOAD_TRAIN_SET: lambda args: print_load_df_event_args(args, elapsed_time),
        ON_END_METRIC:
            lambda args: print('Finished metric', args.metric_config.name,
                               f'in {elapsed_time:1.4f}s'),
    }
