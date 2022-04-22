"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from rexmex.metrics.ranking \
    import normalized_discounted_cumulative_gain, \
    average_precision_at_k, \
    average_recall_at_k, mean_reciprocal_rank
from rexmex.metrics.rating import root_mean_squared_error, mean_absolute_error
from rexmex.metrics.coverage import item_coverage, user_coverage
from rexmex.metrics import intra_list_similarity, novelty
from fairreckitlib.metrics.evaluator import Evaluator
from fairreckitlib.metrics.common import Metric


class EvaluatorRexmex(Evaluator):
    """
    Evaluates results using Rexmex library metrics
    """

    metricDict = {
        Metric.NDCG: normalized_discounted_cumulative_gain,
        Metric.PRECISION: average_precision_at_k,
        Metric.RECALL: average_recall_at_k,
        #Metric.mrr: mean_reciprocal_rank,
        Metric.RMSE: root_mean_squared_error,
        Metric.MAE: mean_absolute_error,
        Metric.ITEM_COVERAGE: item_coverage,
        Metric.USER_COVERAGE: user_coverage,
        Metric.INTRA_LIST_SIMILARITY: intra_list_similarity,
        Metric.NOVELTY: novelty
    }

    def load_test(self, test_path):
        # TODO these header names are arbitrary
        header_names = ["source_id", "target_id", "y_true"]
        return pd.read_csv(test_path, header=None, sep='\t', names=header_names)

    def load_train(self, train_path):
        header_names = ["source_id", "target_id", "y_true"]
        return pd.read_csv(train_path, header=None, sep='\t', names=header_names)

    def load_recs(self, recs_path):
        header_names = ["source_id", "target_id", "y_score"]
        return pd.read_csv(recs_path, header=None, sep='\t', names=header_names)

    def evaluate(self):
        # evaluations = []
        # for metric in self.metrics:
        # TODO refactor self.metrics to metric?
        (metric, k) = self.metrics[0]
        eval_func = EvaluatorRexmex.metricDict[metric]
        print('Debug |', eval_func)
        # TODO refactor
        if metric == Metric.NDCG:
            # TODO needs a specific (multiclass-multi..) format
            raise NotImplementedError
        if metric in [Metric.RMSE, Metric.MAE]:
            # Merge on user ID
            # TODO used in Lenskit as well, refactor?
            scores = pd.merge(self.test, self.recs,  how='left', on=['source_id', 'target_id'])
            print(scores.head())
            evaluation = eval_func(scores['y_true'], scores['y_score'])
        elif metric in [Metric.PRECISION, Metric.RECALL]:
            #recs = self.recs.sort_values(['source_id','y_score'],ascending=False).groupBy('source_id') # Make ordered list of recs
            # assuming recs are already sorted
            evaluation = eval_func(self.test['target_id'], self.recs['target_id'], k)
        #elif metric in [Metric.mrr]:
            # TODO error relevant item did not appear in recommendation
            #raise NotImplementedError
            #evaluation = eval_func(self.test['target_id'], self.recs['target_id'])
        elif metric in [Metric.ITEM_COVERAGE, Metric.USER_COVERAGE]:
            possible_users_items, tuple_recs = self.prepare_for_coverage()
            evaluation = eval_func(possible_users_items, tuple_recs)
        elif metric == Metric.INTRA_LIST_SIMILARITY:
            # eval_func()
            # TODO needs a weird matrix thing
            raise NotImplementedError
        elif metric == Metric.NOVELTY:
            # TODO needs item_popularities
            raise NotImplementedError

        return evaluation
        # evaluations.append(evaluation)

        # return evaluations

    def prepare_for_coverage(self):
        """
        Uses the train set and recommendations set to
        create a tuple that describes the possible user-item pairs in the train set
        and the user-item pairs in the result

        :return: a tuple containing (A,B) where
            a is a tuple (List,List) of possible users and items in the train set and
            b is a list of user-item tuples in the recommendations
        """
        # Convert recommended user, item columns to list of tuples.
        tuple_recs = [tuple(r) for r in self.recs[['source_id', 'target_id']].to_numpy()]
        print(tuple_recs[0:10])
        # The possible users and items are in the train set
        possible_users_items = (self.train['source_id'], self.train['target_id'])
        print(possible_users_items)
        return possible_users_items, tuple_recs
