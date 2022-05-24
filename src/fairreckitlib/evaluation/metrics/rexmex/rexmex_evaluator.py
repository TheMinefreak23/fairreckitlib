"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from typing import Any, Dict

import pandas as pd
from lenskit.metrics import predict

from src.fairreckitlib.evaluation.metrics.evaluator import Evaluator


class RexmexEvaluator(Evaluator):
    """Predictor implementation for the LensKit framework."""

    # TODO eval_func is a function
    def __init__(self, eval_func: Any, params: Dict[str, Any], **kwargs):
        Evaluator.__init__(self, eval_func, params, **kwargs)

    def evaluate(self, train_set, test_set, recs):
        # Coverage metrics
        #elif metric in [Metric.ITEM_COVERAGE, Metric.USER_COVERAGE]:
        possible_users_items, tuple_recs = self.prepare_for_coverage(train_set, recs)
        evaluation = self.eval_func(possible_users_items, tuple_recs)
        #evaluation = self.eval_func(self.params['possible_users_items'], self.params[tuple_recs])

        """
        elif metric == Metric.INTRA_LIST_SIMILARITY:
        # eval_func()
        # TODO needs a weird matrix thing
        raise NotImplementedError
    
        elif metric == Metric.NOVELTY:
        # TODO needs item_popularities
        raise NotImplementedError
        
        """
        return evaluation


    def prepare_for_coverage(self, train_set, recs):
        """
        Uses the train set and recommendations set to
        create a tuple that describes the possible user-item pairs in the train set
        and the user-item pairs in the result

        Returns: a tuple containing (A,B) where
            a is a tuple (List,List) of possible users and items in the train set and
            b is a list of user-item tuples in the recommendations
        """
        # Convert recommended user, item columns to list of tuples.
        tuple_recs = [tuple(r) for r in recs[['user', 'item']].to_numpy()]
        print(tuple_recs[0:10])
        # The possible users and items are in the train set
        possible_users_items = (train_set['user'], train_set['item'])
        print(possible_users_items)
        return possible_users_items, tuple_recs
