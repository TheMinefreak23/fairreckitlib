"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from datetime import datetime
import os
import time

import json
from typing import Tuple

import pandas as pd

from .evaluation_config import MetricConfig
from .evaluation_event import ON_BEGIN_LOAD_TRAIN_SET, ON_END_LOAD_TRAIN_SET, ON_BEGIN_LOAD_TEST_SET, \
    ON_END_LOAD_TEST_SET, ON_BEGIN_LOAD_RECS_SET, ON_END_LOAD_RECS_SET
from ..metrics.evaluator import Evaluator
from ...core.event_dispatcher import EventDispatcher
from ...core.event_error import ON_RAISE_ERROR
from ...core.event_io import ON_MAKE_DIR, ON_REMOVE_FILE
from ...core.factories import Factory
from ...evaluation.pipeline import evaluation_event
from ..metrics.common import metric_matches_type
from src.fairreckitlib.evaluation.metrics.lenskit.lenskit_evaluator import LensKitEvaluator
from src.fairreckitlib.evaluation.metrics.rexmex.rexmex_evaluator import EvaluatorRexmex
from ..metrics.filter import filter_data


class EvaluationPipeline:
    # test
    test_filter = False
    test_use_lenskit = True

    def __init__(self, metric_factory: Factory, event_dispatcher: EventDispatcher):
        self.metric_factory = metric_factory

        # self.train_set = None
        # self.test = None
        # self.recs = None

        self.event_dispatcher = event_dispatcher

    def run(self, recs_path, data_transition, eval_config, is_running, **kwargs):
        # self.profile_path = profile_path
        # self.metrics = metrics
        # self.k = k
        # self.filters = filters

        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_EVAL_PIPELINE,
            # num_metrics=len(self.metrics)
        )
        start = time.time()
        print('eval_config', eval_config)
        for metric in eval_config:
            evaluator = self.metric_factory.create(
                metric.name,
                metric.params,
                **kwargs
            )
            print('data_transition', data_transition)
            self.filter(evaluator, metric, data_transition.train_set_path, data_transition.test_set_path, recs_path)

        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_EVAL_PIPELINE,
            # num_metrics=len(self.metrics),
            elapsed_time=time.time() - start
        )

    def filter(self, evaluator, eval_config, train_path, test_path, recs_path, profile_path=""):

        # Run evaluation globally
        train_set, test_set, recs_set = self.load_data(train_path, test_path, recs_path)
        self.run_evaluation(evaluator, eval_config, train_set, test_set, recs_set, recs_path)

        if self.test_filter:
            self.event_dispatcher.dispatch(
                evaluation_event.ON_BEGIN_FILTER,
                # num_metrics=len(self.metrics)
            )
            filter_start = time.time()
            print('TODO: filter data per evaluation')

            # TODO
            """
            suffix = 'filtered'
            for filter_passes in self.filters:
                # Make temporary filtered data
                filtered_paths = []
                for path in [train_path, test_path, recs_path]:
                    df = pd.read_csv(path, header=None, sep='\t', names=['user', 'item', 'rating'])  # TODO refactor
                    profile = pd.read_csv(profile_path, header=None, sep='\t',
                                          names=['user', 'gender', 'age', 'country', 'date'])
                    merged = df.merge(df, profile, on=['user'])
                    print(merged.head())
                    df = filter_data(merged, filter_passes)['user', 'item', 'rating']
                    print(df.head())
                    filtered_path = path + suffix
                    filtered_paths.append(filtered_path)
                    pd.write_csv(df, filtered_path, header=None, sep='\t')

                filtered_train_set, filtered_test_set, filtered_recs = self.load_data(train_path + suffix,
                                                                                      test_path + suffix,
                                                                                      recs_path + suffix)   # TODO perhaps no need to filter the train/recs, but might be faster

                # Run evaluation per filtered result
                self.run_evaluation(filtered_train_set, filtered_test_set, filtered_recs)

                # Remove filtered data
                for path in filtered_path:
                    os.remove(path)

                    self.event_dispatcher.dispatch(
                        ON_REMOVE_FILE,
                        file=path
                    )
"""
            self.event_dispatcher.dispatch(
                evaluation_event.ON_END_FILTER,
                elapsed_time=time.time() - filter_start
            )

    # TODO refactor so it take dataframes instead of path?
    def run_evaluation(self, evaluator: Evaluator, eval_config: MetricConfig, train_set, test_set, recs, recs_path):

        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_EVAL,
            metric_name=eval_config.name
        )
        start = time.time()
        evaluation = evaluator.evaluate(test_set, recs)
        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_EVAL,
            metric_name=eval_config.name,
            elapsed_time=time.time() - start
        )

        # Create evaluations file.
        file_path = os.path.dirname(recs_path) + "/evaluations.json"
        with open(file_path, mode='w', encoding='utf-8') as out_file:
            json.dump({'evaluations': []}, out_file, indent=4)

        self.event_dispatcher.dispatch(
            ON_MAKE_DIR,
            dir=file_path
        )

        """
        evaluations = []

        for metric in self.metrics:
            if not metric_matches_type(metric,self.rec_type):
                print('Debug | WARNING: Type of metric ' + metric.value + ' doesn\'t match type of data, skipping..')
                continue
            if self.test_use_lenskit and metric in LensKitEvaluator.metric_dict.keys():
                print('Debug | Lenskit:') # TODO CALLBACK
                evaluator = LensKitEvaluator(train_path=train_path,
                                             test_path=test_path,
                                             recs_path=recs_path,
                                             metrics=[(metric, self.k)],
                                             event_dispatcher=self.event_dispatcher)
            elif metric in EvaluatorRexmex.metric_dict.keys():
                print('Debug | Rexmex:') # TODO CALLBACK
                evaluator = EvaluatorRexmex(train_path=train_path,
                                            test_path=test_path,
                                            recs_path=recs_path,
                                            metrics=[(metric, self.k)],
                                            event_dispatcher=self.event_dispatcher)
            else:
                print('Debug | Metric not supported.')
                continue

            evaluation = evaluator.evaluate_process()
            print(evaluation)
            evaluations.append({metric.value: evaluation})

        #print(evaluations)
        """

        self.add_evaluation_to_file(file_path, evaluation, eval_config)

    def add_evaluation_to_file(self, file_path, evaluation_value, eval_config):
        # TODO filters
        evaluation = {'name': eval_config.name,
                      'params': eval_config.params,
                      'evaluation': {'global': evaluation_value, 'filtered': []}}

        # TODO refactor
        with open(file_path) as out_file:
            evaluations = json.load(out_file)

        evaluations['evaluations'].append(evaluation)

        with open(file_path, mode='w') as out_file:
            json.dump(evaluations, out_file, indent=4)

    def load_data(self, train_set_path: str, test_set_path: str, recs_path: str) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.load_train_set(train_set_path), self.load_test_set(test_set_path), self.load_recs(recs_path)

    # TODO EXCEPT FOR RETURN VALUE THIS IS A COPY PASTE FROM MODEL PIPELINE!!!!!!!!!
    def load_train_set(self, train_set_path: str) -> pd.DataFrame:
        """Load the train set that all models can use for training.

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
        """Load the test set that all models can use for testing.

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
            """
            recs = pd.read_csv(
                recs_path,
                sep='\t',
                header=None,
                names=['user', 'item', 'rating']
            )
            """
            # recs = pd.read_csv(recs_path, header=None, sep='\t', names=['user', 'item', 'score'])
            # recs['rank'] = recs.groupby('user')['score'].rank()
            recs = pd.read_csv(recs_path, sep='\t')

            # Reorder/filter columns for LensKit format
            recs = recs[['item', 'score', 'user', 'rank']]

            # LensKit needs this column, TODO refactor?
            # It is a bit redundant because there is only one approach during the evaluation pipeline
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
