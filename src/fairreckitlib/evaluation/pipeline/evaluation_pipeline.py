"""This module contains base functionality of the complete evaluation pipeline.

Classes:

    EvaluationPipeline: class that runs multiple metric computations for a specific evaluation set.

Functions:

    add_evaluation_to_file: append computed evaluation to a result file.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time
from typing import Callable, List, Optional, Tuple


from ...core.config.config_factories import Factory, GroupFactory, resolve_factory
from ...core.core_constants import KEY_NAME, KEY_PARAMS
from ...core.events.event_dispatcher import EventDispatcher
from ...core.events.event_error import ON_FAILURE_ERROR, ON_RAISE_ERROR, ErrorEventArgs
from ...core.io.io_create import create_json
from ...core.io.io_utility import load_json, save_json
from ...core.pipeline.core_pipeline import CorePipeline
from ...data.filter.filter_config import DataSubsetConfig
from ...data.filter.filter_event import FilterDataframeEventArgs
from ...data.filter.filter_passes import filter_from_filter_passes
from ...data.set.dataset import Dataset
from ..metrics.metric_base import BaseMetric
from ..metrics.metric_constants import KEY_METRIC_EVALUATION, KEY_METRIC_SUBGROUP
from ..evaluation_sets import EvaluationSetPaths, EvaluationSets
from .evaluation_config import MetricConfig
from .evaluation_event import EvaluationPipelineEventArgs, MetricEventArgs
from .evaluation_event import ON_BEGIN_EVAL_PIPELINE, ON_END_EVAL_PIPELINE
from .evaluation_event import ON_BEGIN_EVAL_METRIC, ON_END_EVAL_METRIC
from .evaluation_event import ON_BEGIN_FILTER_RECS, ON_END_FILTER_RECS
from .evaluation_event import ON_BEGIN_LOAD_TRAIN_SET, ON_END_LOAD_TRAIN_SET
from .evaluation_event import ON_BEGIN_LOAD_TEST_SET, ON_END_LOAD_TEST_SET
from .evaluation_event import ON_BEGIN_LOAD_RATING_SET, ON_END_LOAD_RATING_SET
from .evaluation_event import ON_BEGIN_METRIC, ON_END_METRIC


class EvaluationPipeline(CorePipeline):
    """Evaluation Pipeline to run metric computations for ratings related to a dataset.

    The pipeline is intended to be used multiple times on different computed rating files
    that are all associated to a specific dataset.
    Loading the evaluation sets is done each time for every metric configuration, so that
    they can be filtered individually on subgroups before computing the actual evaluation.
    For each metric it executes the following steps:

    1) create the metric.
    2) load the evaluation sets.
    3) filter the evaluation sets (optional).
    4) compute the evaluation of the metric.
    5) store evaluations in an overview file.

    Public methods:

    run
    """

    def __init__(
            self,
            dataset: Dataset,
            data_filter_factory: GroupFactory,
            metric_category_factory: GroupFactory,
            event_dispatcher: EventDispatcher):
        """Construct the evaluation pipeline.

        Args:
            dataset: the dataset that is associated with the evaluation of the rating sets.
            data_filter_factory: the factory with available filters for all dataset-matrix pairs.
            metric_category_factory: the metric category factory with available metric factories.
            event_dispatcher: used to dispatch model/IO events when running the pipeline.
        """
        CorePipeline.__init__(self, event_dispatcher)
        self.dataset = dataset
        self.data_filter_factory = data_filter_factory
        self.metric_category_factory = metric_category_factory

    def run(self,
            output_dir: str,
            eval_set_paths: EvaluationSetPaths,
            metric_config_list: List[MetricConfig],
            is_running: Callable[[], bool],
            **kwargs) -> None:
        """Run the entire pipeline from beginning to end.

        Effectively running all computations of the specified metrics.
        All the specified metric configurations that have a subgroup are expected
        to be related to the dataset that was used to construct the pipeline.

        Args:
            output_path: the path of the json file to store the output.
            eval_set_paths: the file paths of the evaluation sets.
            metric_config_list: list of MetricConfig objects to compute.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            reserved for future use
        """
        self.event_dispatcher.dispatch(EvaluationPipelineEventArgs(
            ON_BEGIN_EVAL_PIPELINE,
            metric_config_list
        ))

        start = time.time()
        output_path = os.path.join(output_dir, 'evaluations.json')

        # Create evaluations file
        create_json(
            output_path,
            {'evaluations': []},
            self.event_dispatcher,
            indent=4
        )

        for metric_config in metric_config_list:
            if not is_running():
                return

            metric_factory = resolve_factory(
                metric_config.name,
                self.metric_category_factory
            )
            if metric_factory is None:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_FAILURE_ERROR,
                    'Failure: to resolve metric factory for metric \'' + metric_config.name + '\''
                ))
                continue

            try:
                self.run_metric(
                    (output_dir, output_path),
                    metric_factory,
                    eval_set_paths,
                    metric_config,
                    **kwargs
                )
            except ArithmeticError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'ArithmeticError: trying to run metric ' + metric_config.name
                ))
                continue
            except FileNotFoundError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'FileNotFoundError: trying to run metric ' + metric_config.name
                ))
                continue
            except MemoryError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'MemoryError: trying to run metric ' + metric_config.name
                ))
                continue
            except RuntimeError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'RuntimeError: trying to run metric ' + metric_config.name
                ))
                continue

        end = time.time()

        self.event_dispatcher.dispatch(EvaluationPipelineEventArgs(
            ON_END_EVAL_PIPELINE,
            metric_config_list
        ), elapsed_time=end - start)

    def run_metric(
            self,
            output_paths: Tuple[str, str],
            metric_factory: Factory,
            eval_set_paths: EvaluationSetPaths,
            metric_config: MetricConfig,
            **kwargs) -> None:
        """Run the evaluation computation for the specified metric configuration.

        Args:
            output_path: the path of the json file to store the output.
            metric_factory: the factory that contains the specified metric.
            eval_set_paths: the file paths of the evaluation sets.
            metric_config: the metric evaluation configuration.

        Raises:
            ArithmeticError: possibly raised by a metric on construction or evaluation.
            MemoryError: possibly raised by a metric on construction or evaluation.
            RuntimeError: possibly raised by a metric on construction or evaluation.
            FileNotFoundError: when the file of one of the evaluation sets does not exist.

        Keyword Args:
            reserved for future use
        """
        self.event_dispatcher.dispatch(MetricEventArgs(
            ON_BEGIN_METRIC,
            metric_config
        ))

        [output_dir, output_path] = output_paths

        start = time.time()

        metric = metric_factory.create(
            metric_config.name,
            metric_config.params,
            **kwargs
        )

        # this can raise a FileNotFoundError, effectively aborting the pipeline
        eval_sets = self.load_evaluation_sets(
            eval_set_paths,
            metric.requires_train_set,
            metric.requires_test_set
        )

        eval_sets = self.filter_set_rows(
            output_dir,
            eval_sets,
            metric_config.subgroup
        )

        evaluation = self.compute_metric_evaluation(
            metric,
            eval_sets,
            metric_config
        )

        add_evaluation_to_file(output_path, evaluation, metric_config)

        end = time.time()

        self.event_dispatcher.dispatch(MetricEventArgs(
            ON_END_METRIC,
            metric_config
        ), elapsed_time=end - start)

    def load_evaluation_sets(
            self,
            eval_set_paths: EvaluationSetPaths,
            train_set_required: bool,
            test_set_required: bool) -> EvaluationSets:
        """Load the required evaluation sets.

        Args:
            eval_set_paths: the file paths of the evaluation sets.
            train_set_required: whether the train set is required for the evaluation.
            test_set_required: whether the test set is required for the evaluation.

        Raises:
            FileNotFoundError: when the file of one of the evaluation sets does not exist.

        Returns:
            the loaded evaluation sets.
        """
        rating_set = self.read_dataframe(
            eval_set_paths.ratings_path,
            'rating set',
            ON_BEGIN_LOAD_RATING_SET,
            ON_END_LOAD_RATING_SET,
        )

        train_set = None if not train_set_required else self.read_dataframe(
            eval_set_paths.train_path,
            'train set',
            ON_BEGIN_LOAD_TRAIN_SET,
            ON_END_LOAD_TRAIN_SET,
            names=['user', 'item', 'rating']
        )

        test_set = None if not test_set_required else self.read_dataframe(
            eval_set_paths.test_path,
            'test set',
            ON_BEGIN_LOAD_TEST_SET,
            ON_END_LOAD_TEST_SET,
            names=['user', 'item', 'rating']
        )

        return EvaluationSets(rating_set, train_set, test_set)

    def filter_set_rows(
            self,
            output_dir,
            eval_sets: EvaluationSets,
            subgroup: Optional[DataSubsetConfig]) -> EvaluationSets:
        """Filter the evaluation set rows for the specified subgroup.

        The subset is created by applying multiple filter passes to the evaluation sets
        individually. These filter passes are then combined to form the resulting sets.

        Args:
            eval_sets: the evaluation sets to filter.
            subgroup: the subgroup to create of the evaluation sets.

        Returns:
            the filtered evaluation sets.
        """
        # early exit, because no filtering is needed
        if subgroup is None or len(subgroup.filter_passes) == 0:
            return eval_sets

        self.event_dispatcher.dispatch(FilterDataframeEventArgs(
            ON_BEGIN_FILTER_RECS,
            subgroup
        ))

        start = time.time()

        # Filter for each dataframe in eval_sets
        filter_factory = self.data_filter_factory
        if eval_sets.train is not None:
            eval_sets.train = filter_from_filter_passes(
                    self, output_dir, eval_sets.train, subgroup, filter_factory)
        if eval_sets.test is not None:
            eval_sets.test = filter_from_filter_passes(
                self, output_dir, eval_sets.test, subgroup, filter_factory)
        eval_sets.ratings = filter_from_filter_passes(
                self, output_dir, eval_sets.ratings, subgroup, filter_factory)
        end = time.time()
        self.event_dispatcher.dispatch(FilterDataframeEventArgs(
            ON_END_FILTER_RECS,
            subgroup
        ), elapsed_time=end - start)

        return eval_sets

    def compute_metric_evaluation(
            self,
            metric: BaseMetric,
            eval_sets: EvaluationSets,
            metric_config: MetricConfig) -> float:
        """Compute the evaluation for the specified metric on the specified sets.

        Args:
            metric: the metric to use for computing the evaluation.
            eval_sets: the evaluation sets to compute the performance of.
            metric_config: the metric configuration that is associated with the metric.

        Raises:
            ArithmeticError: possibly raised by a metric on evaluation.
            MemoryError: possibly raised by a metric on evaluation.
            RuntimeError: possibly raised by a metric on evaluation.

        Returns:
            the computed evaluation of the metric.
        """
        self.event_dispatcher.dispatch(MetricEventArgs(
            ON_BEGIN_EVAL_METRIC,
            metric_config
        ))

        start = time.time()

        evaluation = metric.evaluate(eval_sets)

        end = time.time()

        self.event_dispatcher.dispatch(MetricEventArgs(
            ON_END_EVAL_METRIC,
            metric_config
        ), elapsed_time=end - start)

        return evaluation


def add_evaluation_to_file(
        file_path: str,
        evaluation_value: float,
        metric_config: MetricConfig) -> None:
    """Add an evaluation result to the list in the overview file.

    Args:
        file_path: the path to the evaluations overview file.
        evaluation_value: the evaluation result.
        metric_config: the metric configuration used for the evaluation.
    """
    subgroup = {} if metric_config.subgroup is None else metric_config.subgroup.to_yml_format()
    evaluation = {KEY_NAME: metric_config.name,
                  KEY_PARAMS: metric_config.params,
                  KEY_METRIC_EVALUATION:{'value': evaluation_value, KEY_METRIC_SUBGROUP: subgroup}}

    evaluations = load_json(file_path)
    evaluations['evaluations'].append(evaluation)
    save_json(file_path, evaluations, indent=4)
