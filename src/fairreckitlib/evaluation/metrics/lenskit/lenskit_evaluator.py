"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from lenskit import topn
from lenskit.metrics import predict

from src.fairreckitlib.evaluation.metrics.evaluator import Evaluator
from src.fairreckitlib.evaluation.metrics.common import Metric


class LensKitEvaluator(Evaluator):
    """
    Evaluates results using LensKit library metrics
    """

    metric_dict = {
        Metric.NDCG: topn.ndcg,
        Metric.PRECISION: topn.precision,
        Metric.RECALL: topn.recall,
        Metric.MRR: topn.recip_rank,

        Metric.RMSE: predict.rmse,
        Metric.MAE: predict.mae,
    }

    # TODO refactor
    group_dict = {
        Metric.NDCG: 'ndcg',
        Metric.PRECISION: 'precision',
        Metric.RECALL: 'recall',
        Metric.MRR: 'recip_rank',

        Metric.RMSE: 'rmse',
        Metric.MAE: 'mae'
    }

    topn_metrics = [Metric.NDCG, Metric.PRECISION, Metric.RECALL, Metric.MRR]

    def load_test(self, test_path):
        return pd.read_csv(test_path, header=None, sep='\t', names=['user', 'item', 'rating'])

    def load_train(self, train_path):
        return pd.read_csv(train_path, header=None, sep='\t', names=['user', 'item', 'rating'])

    def load_recs(self, recs_path):
        recs = pd.read_csv(recs_path, header=None, sep='\t', names=['user', 'item', 'score'])
        recs['rank'] = recs.groupby('user')['score'].rank()
        recs['Algorithm'] = 'APPROACHNAME'
        return recs

    def evaluate(self):
        # evaluations = []
        # for metric in self.metrics:
        # TODO refactor self.metrics to metric?
        (metric, k) = self.metrics[0]
        eval_func = LensKitEvaluator.metric_dict[metric]
        print(eval_func)
        if metric in LensKitEvaluator.topn_metrics:
            analysis = topn.RecListAnalysis()
            analysis.add_metric(eval_func, k=k)
            results = analysis.compute(self.recs, self.test).head()

            group_name = LensKitEvaluator.group_dict[metric]
            evaluation = results.groupby('Algorithm')[group_name].mean()[0]
        else:
            # TODO USER VS GLOBAL
            # TODO handle predictions without truth (missing)? Lenskit has 'ignore' or 'error'
            if metric in [Metric.RMSE, Metric.MAE]:
                # Merge on user ID
                scores = pd.merge(self.test, self.recs, how='left', on=['user','item'])
                scores.rename(columns={'score': 'prediction'}, inplace=True)
                print(scores)
                evaluation = predict.user_metric(scores, metric=eval_func)
            else:
                raise Exception # Apparently there is another prediction metric that isn't handled

        return evaluation
        # evaluations.append(results[0])

        # return evaluations
