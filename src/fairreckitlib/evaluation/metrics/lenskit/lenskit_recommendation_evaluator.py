"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from typing import Any, Dict

from lenskit import topn

from src.fairreckitlib.core.apis import LENSKIT_API
from src.fairreckitlib.evaluation.metrics.common import KEY_METRIC_PARAM_K
from src.fairreckitlib.evaluation.metrics.evaluator import Evaluator


class LensKitRecommendationEvaluator(Evaluator):
    """Predictor implementation for the LensKit framework."""

    # TODO eval_func is a function
    def __init__(self, eval_func: Any, params: Dict[str, Any], **kwargs):
        Evaluator.__init__(self, eval_func, params, **kwargs)
        self.group = kwargs['group']

    def evaluate(self, train_set, test_set, recs):
        #print(test_set.head())
        #print(recs.head())
        analysis = topn.RecListAnalysis()
        k = self.params.get(KEY_METRIC_PARAM_K)
        if k:
            analysis.add_metric(self.eval_func, k=k)
        else:
            analysis.add_metric(self.eval_func)

        results = analysis.compute(recs, test_set).head()

        evaluation = results.groupby('Algorithm')[self.group].mean()[0]

        return evaluation

    def get_api_name(self):
        return LENSKIT_API
