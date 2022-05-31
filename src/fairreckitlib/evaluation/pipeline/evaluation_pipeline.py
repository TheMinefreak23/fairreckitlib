"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Tuple

import pandas as pd

from ...core.config.config_factories import Factory
from ...core.events.event_dispatcher import EventDispatcher
from ...core.io.io_delete import delete_file
from ...core.io.io_utility import load_json, save_json
from ...core.pipeline.core_pipeline import CorePipeline
from ...data.filter.filter_event import FilterDataframeEventArgs
from ..metrics.evaluator import Evaluator
from ..metrics.filter import filter_data
from .evaluation_config import MetricConfig
from .evaluation_event import EvaluationPipelineEventArgs, MetricEventArgs
from .evaluation_event import ON_BEGIN_EVAL_PIPELINE, ON_END_EVAL_PIPELINE
from .evaluation_event import ON_BEGIN_FILTER_RECS, ON_END_FILTER_RECS
from .evaluation_event import ON_BEGIN_LOAD_TRAIN_SET, ON_END_LOAD_TRAIN_SET
from .evaluation_event import ON_BEGIN_LOAD_TEST_SET, ON_END_LOAD_TEST_SET
from .evaluation_event import ON_BEGIN_LOAD_RECS_SET, ON_END_LOAD_RECS_SET
from .evaluation_event import ON_BEGIN_METRIC, ON_END_METRIC


FILTER_SUFFIX = 'filtered'
# TODO DEV
USE_FILTER = False


class EvaluationPipeline(CorePipeline):
    """Evaluation Pipeline to run evaluations from a specific metric category."""

    def __init__(self, metric_factory: Factory, event_dispatcher: EventDispatcher):
        CorePipeline.__init__(self, event_dispatcher)
        self.metric_factory = metric_factory

    # TODO documentation
    def run(self, out_path: str, recs_path: str, data_transition, metrics, **kwargs):
        """

        Args:
            out_path:
            recs_path:
            data_transition:
            metrics:
            kwargs:
        """
        self.event_dispatcher.dispatch(EvaluationPipelineEventArgs(
            ON_BEGIN_EVAL_PIPELINE,
            metrics
        ))

        start = time.time()

        for metric in metrics:
            # print('metric',metric)
            evaluator = self.metric_factory.create(
                metric.name,
                metric.params,
                **kwargs
            )
            # print('data_transition', data_transition)
            self.filter(evaluator,
                        metric,
                        set_paths=(data_transition.train_set_path,
                                   data_transition.test_set_path,
                                   recs_path),
                        out_path=out_path,
                        profile=data_transition.dataset)

        self.event_dispatcher.dispatch(EvaluationPipelineEventArgs(
            ON_END_EVAL_PIPELINE,
            metrics
        ), elapsed_time=time.time() - start)

    # TODO documentation
    def filter(self, evaluator, metric, *, set_paths, out_path, profile):
        """Run the evaluation on the non-filtered and filtered data
            Args:
        """
        # Run evaluation globally
        sets = self.load_data(set_paths)
        self.run_evaluation(evaluator,
                            metric,
                            sets=sets,
                            out_path=out_path
                            )

        if len(metric.prefilters) == 0:
            return

        self.event_dispatcher.dispatch(FilterDataframeEventArgs(
            ON_BEGIN_FILTER_RECS,
            metric.prefilters
        ))
        filter_start = time.time()
        print('TODO: filter data per evaluation')

        # TODO filter
        if USE_FILTER:
            for filter_passes in metric.prefilters:
                filtered_paths = filter_pass(set_paths,
                                             profile,
                                             filter_passes
                                             )

                sets = self.load_data(set_paths, use_filter=True)

                # Run evaluation per filtered result
                self.run_evaluation(evaluator,
                                    metric,
                                    sets=sets,
                                    out_path=out_path)

                # Remove filtered data
                for path in filtered_paths:
                    delete_file(path, self.event_dispatcher)

            self.event_dispatcher.dispatch(FilterDataframeEventArgs(
                ON_END_FILTER_RECS,
                metric.prefilters
            ), elapsed_time=time.time() - filter_start)

    # TODO documentation
    def run_evaluation(self,
                       evaluator: Evaluator,
                       metric_config: MetricConfig,
                       *,
                       sets,
                       out_path):
        """Run the evaluation for the specified metric configuration.

        Args:
           evaluator:
           metric_config:

        Keyword Args:
            sets:
            out_path:

        """
        self.event_dispatcher.dispatch(MetricEventArgs(
            ON_BEGIN_METRIC,
            metric_config
        ))
        start = time.time()
        train_set, test_set, recs = sets
        evaluation = evaluator.evaluate(train_set, test_set, recs)
        end = time.time()
        self.event_dispatcher.dispatch(MetricEventArgs(
            ON_END_METRIC,
            metric_config
        ), elapsed_time=end - start)

        add_evaluation_to_file(out_path, evaluation, metric_config)

    def load_data(self, set_paths: Tuple[str,str,str], *, use_filter=False) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train and test set, as well as the model recommendations/predictions.

        Returns:
            the train set, test set and model results set (DataFrames)
        """
        if use_filter: # Append filter suffix to path for temporary filtered data storage
            set_paths = [path + FILTER_SUFFIX for path in set_paths]
        train_set_path, test_set_path, recs_path = set_paths
        train_set = self.read_dataframe(
            train_set_path,
            'train set',
            ON_BEGIN_LOAD_TRAIN_SET,
            ON_END_LOAD_TRAIN_SET,
            names=['user', 'item', 'rating']
        )
        test_set = self.read_dataframe(
            test_set_path,
            'test set',
            ON_BEGIN_LOAD_TEST_SET,
            ON_END_LOAD_TEST_SET,
            names=['user', 'item', 'rating']
        )
        recs = self.read_dataframe(
            recs_path,
            'rec set',
            ON_BEGIN_LOAD_RECS_SET,
            ON_END_LOAD_RECS_SET,
        )

        # Reorder/filter columns for LensKit format
        recs = recs[['item', 'score', 'user', 'rank']]

        # LensKit needs this column, TODO refactor?
        # It is a bit redundant because
        # there is only one approach during the evaluation pipeline
        recs['Algorithm'] = 'APPROACHNAME'

        return train_set, test_set, recs


def add_evaluation_to_file(file_path, evaluation_value, metric_config):
    """Add an evaluation result to the list in the overview file.

    Args:
        file_path: the path to the evaluations overview file.
        evaluation_value: the evaluation result.
        metric_config: the metric configuration used for the evaluation.
    """
    # TODO filters
    evaluation = {'name': metric_config.name,
                  'params': metric_config.params,
                  'evaluation': {'global': evaluation_value, 'filtered': []}}

    # TODO refactor
    evaluations = load_json(file_path)

    evaluations['evaluations'].append(evaluation)
    # print(json.dumps(evaluations, indent=4))

    save_json(file_path, evaluations, indent=4)


def filter_pass(set_paths, profile, filter_passes):
    """Make temporary filtered data

    Args:
        set_paths: paths to the train and test set and model result
        profile: additional dataset information for filtering
        filter_passes: list of filter passes to perform

    Returns:
        the paths of the filtered data.
    """
    filtered_paths = []
    for path in set_paths:
        raw_df = pd.read_csv(
            path,
            header=None,
            sep='\t',
            names=['user', 'item', 'rating']
        )
        merged = raw_df.merge(raw_df, profile, on=['user'])
        print(merged.head())
        filtered_df = filter_data(merged, filter_passes)['user', 'item', 'rating']
        print(filtered_df.head())
        filtered_path = path + FILTER_SUFFIX
        filtered_paths.append(filtered_path)
        pd.write_csv(filtered_df, filtered_path, header=None, sep='\t')

    return filtered_paths
