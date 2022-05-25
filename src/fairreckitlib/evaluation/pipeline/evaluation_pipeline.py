"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os
import time

import json
from typing import Tuple

import pandas as pd

from .evaluation_config import MetricConfig
from .evaluation_event import ON_BEGIN_LOAD_TRAIN_SET, \
    ON_END_LOAD_TRAIN_SET, ON_BEGIN_LOAD_TEST_SET, \
    ON_END_LOAD_TEST_SET, ON_BEGIN_LOAD_RECS_SET, ON_END_LOAD_RECS_SET
from ..metrics.evaluator import Evaluator
from ..metrics.filter import filter_data
from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_RAISE_ERROR
from ...core.event_io import ON_REMOVE_FILE
from ...core.factories import Factory
from ...evaluation.pipeline import evaluation_event

FILTER_SUFFIX = 'filtered'
# TODO DEV
USE_FILTER = False


class EvaluationPipeline:
    """Evaluation Pipeline to run evaluations from a specific metric category."""

    def __init__(self, metric_factory: Factory, event_dispatcher: EventDispatcher):
        self.metric_factory = metric_factory
        self.filters = []  # TODO

        self.event_dispatcher = event_dispatcher

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
        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_EVAL_PIPELINE,
            num_metrics=len(metrics)
        )
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

        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_EVAL_PIPELINE,
            num_metrics=len(metrics),
            elapsed_time=time.time() - start
        )

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

        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_FILTER,
            filter_name=''#TODO
        )
        filter_start = time.time()
        print('TODO: filter data per evaluation')

        # TODO filter
        if USE_FILTER:
            for filter_passes in self.filters:
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
                    os.remove(path)

                    self.event_dispatcher.dispatch(
                        ON_REMOVE_FILE,
                        file=path
                    )

            self.event_dispatcher.dispatch(
                evaluation_event.ON_END_FILTER,
                elapsed_time=time.time() - filter_start,
                filter_name=''#TODO
            )

    # TODO documentation
    def run_evaluation(self,
                       evaluator: Evaluator,
                       eval_config: MetricConfig,
                       *,
                       sets,
                       out_path):
        """Run the evaluation for the specified metric configuration.

        Args:
           evaluator:
           eval_config:

        Keyword Args:
            sets:
            out_path:

        """
        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_EVAL,
            metric_name=eval_config.name
        )
        start = time.time()
        train_set, test_set, recs = sets
        evaluation = evaluator.evaluate(train_set, test_set, recs)
        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_EVAL,
            metric_name=eval_config.name,
            elapsed_time=time.time() - start
        )

        add_evaluation_to_file(out_path, evaluation, eval_config)

    def load_data(self, set_paths: Tuple[str,str,str], *, use_filter=False) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train and test set, as well as the model recommendations/predictions.

        Returns:
            the train set, test set and model results set (DataFrames)
        """
        if use_filter: # Append filter suffix to path for temporary filtered data storage
            set_paths = [path + FILTER_SUFFIX for path in set_paths]
        train_set_path, test_set_path, recs_path = set_paths
        train_set = self.load_train_set(train_set_path)
        test_set = self.load_test_set(test_set_path)
        recs = self.load_recs(recs_path)
        return train_set, test_set, recs

    # TODO EXCEPT FOR RETURN VALUE THIS IS A COPY PASTE FROM MODEL PIPELINE!!!!!!!!!
    def load_train_set(self, train_set_path: str) -> pd.DataFrame:
        """Load the train set that the models used for training.

        Args:
            train_set_path: path to where the train set is stored.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_LOAD_TRAIN_SET,
            train_set_path=train_set_path
        )

        start = time.time()

        try:
            train_set = pd.read_csv(
                train_set_path,
                sep='\t',
                header=None,
                names=['user', 'item', 'rating']
            )
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='FileNotFoundError: raised while trying to load the train set from ' +
                    train_set_path
            )
            # raise again so that the pipeline aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_TRAIN_SET,
            train_set_path=train_set_path,
            train_set=train_set,
            elapsed_time=end - start
        )

        return train_set

    def load_test_set(self, test_set_path: str) -> pd.DataFrame:
        """Load the test set that the models used for testing.

        Args:
            test_set_path: path to where the test set is stored.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_LOAD_TEST_SET,
            test_set_path=test_set_path
        )

        start = time.time()

        try:
            test_set = pd.read_csv(
                test_set_path,
                sep='\t',
                header=None,
                names=['user', 'item', 'rating']
            )
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='FileNotFoundError: raised while trying to load the test set from ' +
                    test_set_path
            )
            # raise again so that the pipeline aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_TEST_SET,
            test_set_path=test_set_path,
            test_set=test_set,
            elapsed_time=end - start
        )

        return test_set

    def load_recs(self, recs_path: str) -> pd.DataFrame:
        """Load the model recs result.

        Args:
            recs_path: path to where the recs are stored.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_LOAD_RECS_SET,
            recs_set_path=recs_path
        )

        start = time.time()

        try:
            # recs = pd.read_csv(recs_path, header=None, sep='\t', names=['user', 'item', 'score'])
            # recs['rank'] = recs.groupby('user')['score'].rank()
            recs = pd.read_csv(recs_path, sep='\t')

            # Reorder/filter columns for LensKit format
            recs = recs[['item', 'score', 'user', 'rank']]

            # LensKit needs this column, TODO refactor?
            # It is a bit redundant because
            # there is only one approach during the evaluation pipeline
            recs['Algorithm'] = 'APPROACHNAME'

        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='FileNotFoundError: raised while trying to load the test set from ' +
                    recs_path
            )
            # raise again so that the pipeline aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_RECS_SET,
            recs_set_path=recs_path,
            recs_set=recs,
            elapsed_time=end - start
        )

        return recs


def add_evaluation_to_file(file_path, evaluation_value, eval_config):
    """Add an evaluation result to the list in the overview file.

    Args:
        file_path: the path to the evaluations overview file
        evaluation_value: the evaluation result
        eval_config: the metric configuration used for the evaluation
    """
    # TODO filters
    evaluation = {'name': eval_config.name,
                  'params': eval_config.params,
                  'evaluation': {'global': evaluation_value, 'filtered': []}}

    # TODO refactor
    with open(file_path, encoding='utf-8') as out_file:
        evaluations = json.load(out_file)

    evaluations['evaluations'].append(evaluation)
    # print(json.dumps(evaluations, indent=4))

    with open(file_path, mode='w', encoding='utf-8') as out_file:
        json.dump(evaluations, out_file, indent=4)


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
