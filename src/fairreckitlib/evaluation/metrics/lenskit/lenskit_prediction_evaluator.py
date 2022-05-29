"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from typing import Any, Dict

import pandas as pd
from lenskit.metrics import predict

from ....core.apis import LENSKIT_API
from ..evaluator import Evaluator


class LensKitPredictionEvaluator(Evaluator):
    """Predictor implementation for the LensKit framework."""

    # TODO eval_func is a function
    def __init__(self, eval_func: Any, params: Dict[str, Any], **kwargs):
        Evaluator.__init__(self, eval_func, params, **kwargs)

    def evaluate(self, train_set, test_set, recs):
        # Merge on user ID
        scores = pd.merge(test_set, recs, how='left', on=['user', 'item'])
        scores.rename(columns={'score': 'prediction'}, inplace=True)
        #print(scores)
        evaluation = predict.user_metric(scores, metric=self.eval_func)

        return evaluation

    def get_api_name(self):
        return LENSKIT_API
