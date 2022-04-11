""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from fairreckitlib.metrics.evaluator2 import Evaluator
from fairreckitlib.metrics.metrics2 import Metric


class EvaluatorRexmex(Evaluator):
    from rexmex.metrics.ranking import normalized_discounted_cumulative_gain, average_precision_at_k, \
        average_recall_at_k, mean_reciprocal_rank
    from rexmex.metrics.rating import root_mean_squared_error, mean_absolute_error
    from rexmex.metrics.coverage import item_coverage, user_coverage
    from rexmex.metrics import intra_list_similarity, novelty

    metricDict = {
        Metric.ndcg: normalized_discounted_cumulative_gain,
        Metric.precision: average_precision_at_k,
        Metric.recall: average_recall_at_k,
        #Metric.mrr: mean_reciprocal_rank,
        Metric.rmse: root_mean_squared_error,
        Metric.mae: mean_absolute_error,
        Metric.item_coverage: item_coverage,
        Metric.user_coverage: user_coverage,
        Metric.intra_list_similarity: intra_list_similarity,
        Metric.novelty: novelty
    }

    def load_test(self, test_path):
        # TODO these header names are arbitrary
        self.test = pd.read_csv(test_path, header=None, sep='\t', names=["source_id", "target_id", "y_true"])

    def load_train(self, train_path):
        self.train = pd.read_csv(train_path, header=None, sep='\t', names=["source_id", "target_id", "y_true"])

    def load_recs(self, recs_path):
        self.recs = pd.read_csv(recs_path, header=None, sep='\t', names=["source_id", "target_id", "y_score"])

    def evaluate(self):
        # evaluations = []
        # for metric in self.metrics:
        # TODO refactor self.metrics to metric?
        (metric, k) = self.metrics[0]
        eval_func = EvaluatorRexmex.metricDict[metric]
        print(eval_func)
        # TODO refactor
        if metric == Metric.ndcg:
            # TODO needs a specific (multiclass-multi..) format
            raise NotImplementedError
        if metric in [Metric.rmse, Metric.mae]:
            # Merge on user ID # TODO used in Lenskit as well, refactor?
            scores = pd.merge(self.test, self.recs,  how='left', on=['source_id', 'target_id'])
            print(scores.head())
            evaluation = eval_func(scores['y_true'], scores['y_score'])
        elif metric in [Metric.precision, Metric.recall]:
            #recs = self.recs.sort_values(['source_id','y_score'],ascending=False).groupBy('source_id') # Make ordered list of recs
            # assuming recs are already sorted
            evaluation = eval_func(self.test['target_id'], self.recs['target_id'], k)
        #elif metric in [Metric.mrr]:
            # TODO error relevant item did not appear in recommendation
            #raise NotImplementedError
            #evaluation = eval_func(self.test['target_id'], self.recs['target_id'])
        elif metric in [Metric.item_coverage, Metric.user_coverage]:
            possible_users_items, tuple_recs = self.prepare_for_coverage()
            evaluation = eval_func(possible_users_items, tuple_recs)
        elif metric == Metric.intra_list_similarity:
            # eval_func()
            # TODO needs a weird matrix thing
            raise NotImplementedError
        elif metric == Metric.novelty:
            # TODO needs item_popularities
            raise NotImplementedError

        return evaluation
        # evaluations.append(evaluation)

        # return evaluations

    def prepare_for_coverage(self):
        # Convert recommended user, item columns to list of tuples.
        tuple_recs = [tuple(r) for r in self.recs[['source_id', 'target_id']].to_numpy()]
        print(tuple_recs[0:10])
        # The possible users and items are in the train set
        possible_users_items = (self.train['source_id'], self.train['target_id'])
        print(possible_users_items)
        return possible_users_items, tuple_recs
