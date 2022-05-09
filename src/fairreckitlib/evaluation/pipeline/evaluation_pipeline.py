"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from datetime import datetime
import os
import time

import json
import pandas as pd

from ...events import evaluation_event, io_event
from ..metrics.evaluator_lenskit import EvaluatorLenskit
from ..metrics.evaluator_rexmex import EvaluatorRexmex
from ..metrics.common import metric_matches_type


class EvaluationPipeline:
    # test
    test_filter = False
    test_use_lenskit = True

    def __init__(self, rec_result, profile_path, metrics, k, filters, event_dispatcher):
        self.name = rec_result.name
        self.train_path = rec_result.train_path
        self.test_path = rec_result.test_path
        self.recs_path = rec_result.recs_path
        self.rec_type = rec_result.rec_type
        self.profile_path = profile_path
        self.metrics = metrics
        self.k = k
        self.filters = filters

        self.event_dispatcher = event_dispatcher

    def run(self):
        self.filter_all()

    def filter_all(self):

        # Run evaluation globally
        self.evaluate_all(self.train_path, self.test_path, self.recs_path)

        if self.test_filter:

            self.event_dispatcher.dispatch(
                evaluation_event.ON_BEGIN_FILTER,
                num_metrics=len(self.metrics)
            )
            filter_start = time.time()

            from filter import filter_data

            suffix = 'filtered'

            for filter_passes in self.filters:
                # Make temporary filtered data
                for path in [self.train_path, self.test_path, self.recs_path]:
                    df = pd.read_csv(path, header=None, sep='\t', names=['user', 'item', 'rating'])  # TODO refactor
                    profile = pd.read_csv(self.profile_path, header=None, sep='\t',
                                          names=['user', 'gender', 'age', 'country', 'date'])
                    merged = df.merge(df, profile, on=['user'])
                    print(merged.head())
                    df = filter_data(merged, filter_passes)['user', 'item', 'rating']
                    print(df.head())
                    pd.write_csv(df, path + suffix, header=None, sep='\t')

                self.train_path = self.train_path + suffix  # TODO perhaps no need to filter the train/recs, but might be faster
                self.test_path = self.test_path + suffix
                self.recs_path = self.recs_path + suffix

                # Run evaluation per filtered result
                self.evaluate_all(self.train_path, self.test_path, self.recs_path)

                for path in [self.train_path, self.test_path, self.recs_path]:
                    import os
                    os.remove(path)

                    self.event_dispatcher.dispatch(
                        io_event.on_remove_file,
                        file=path
                    )

            self.event_dispatcher.dispatch(
                evaluation_event.ON_END_FILTER,
                elapsed_time=time.time()-filter_start
            )

    # TODO refactor so it take dataframes instead of path?
    def evaluate_all(self, train_path, test_path, recs_path):

        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_EVAL_PIPELINE,
            num_metrics=len(self.metrics)
        )
        eval_start = time.time()

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print('Starting evaluation..')

        evaluations = []

        for metric in self.metrics:
            if not metric_matches_type(metric,self.rec_type):
                print('Debug | WARNING: Type of metric ' + metric.value + ' doesn\'t match type of data, skipping..')
                continue
            if self.test_use_lenskit and metric in EvaluatorLenskit.metric_dict.keys():
                print('Debug | Lenskit:') # TODO CALLBACK
                evaluator = EvaluatorLenskit(train_path=train_path,
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

        #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        #print('End of evaluation.')


        #print('Writing evaluations to file..')
        file_path =os.path.dirname(recs_path)+"/evaluations.json"
        out_file = open(file_path, "w")
        json.dump({'evaluations': evaluations}, out_file, indent=4)

        self.event_dispatcher.dispatch(
            io_event.on_make_dir,
            dir=file_path
        )

        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_EVAL_PIPELINE,
            num_metrics=len(self.metrics),
            elapsed_time=time.time()-eval_start
        )