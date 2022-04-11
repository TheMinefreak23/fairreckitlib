from datetime import datetime

import json

import pandas as pd

from fairreckitlib.metrics.evaluator_lenskit import EvaluatorLenskit
from fairreckitlib.metrics.evaluator_rexmex import EvaluatorRexmex
from fairreckitlib.metrics.metrics2 import metric_matches_type


class EvaluationPipeline:
    # test
    test_filter = False
    test_use_lenskit = True

    def __init__(self, rec_result, profile_path, metrics, k, filters):
        self.name = rec_result.name
        self.train_path = rec_result.train_path
        self.test_path = rec_result.test_path
        self.recs_path = rec_result.recs_path
        self.rec_type = rec_result.rec_type
        self.profile_path = profile_path
        self.metrics = metrics
        self.k = k
        self.filters = filters

    def run(self):
        self.filter_all()

    def filter_all(self):

        # Run evaluation globally
        self.evaluate_all(self.train_path, self.test_path, self.recs_path)

        if self.test_filter:
            from filter import filter

            suffix = 'filtered'

            for filter_passes in self.filters:
                # Make temporary filtered data
                for path in [self.train_path, self.test_path, self.recs_path]:
                    df = pd.read_csv(path, header=None, sep='\t', names=['user', 'item', 'rating'])  # TODO refactor
                    profile = pd.read_csv(self.profile_path, header=None, sep='\t',
                                          names=['user', 'gender', 'age', 'country', 'date'])
                    merged = df.merge(df, profile, on=['user'])
                    print(merged.head())
                    df = filter(merged, filter_passes)['user', 'item', 'rating']
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

    # TODO refactor so it take dataframes instead of path?
    def evaluate_all(self, train_path, test_path, recs_path):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print('Starting evaluation..')

        evaluations = []

        for metric in self.metrics:
            if not metric_matches_type(metric,self.rec_type):
                print('WARNING: Type of metric ' + metric.value + ' doesn\'t match type of data, skipping..')
                continue
            if self.test_use_lenskit and metric in EvaluatorLenskit.metricDict.keys():
                print('Lenskit:')
                evaluator = EvaluatorLenskit(train_path=train_path,
                                             test_path=test_path,
                                             recs_path=recs_path,
                                             metrics=[(metric, self.k)])
            elif metric in EvaluatorRexmex.metricDict.keys():
                print('Rexmex:')
                evaluator = EvaluatorRexmex(train_path=train_path,
                                            test_path=test_path,
                                            recs_path=recs_path,
                                            metrics=[(metric, self.k)])
            else:
                print('Metric not supported.')
                continue

            evaluation = evaluator.evaluate()
            print(evaluation)
            evaluations.append({metric.value: evaluation})

        print(evaluations)

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print('End of evaluation.')

        print('Writing evaluations to file..')
        lk_string = '_lk' if self.test_use_lenskit else ''
        out_file = open("evaluations_"+self.name+lk_string+".json", "w")
        json.dump({'evaluations': evaluations}, out_file, indent=4)
        # TODO: write entry in the overview JSON
        print('Saved evaluations.')
