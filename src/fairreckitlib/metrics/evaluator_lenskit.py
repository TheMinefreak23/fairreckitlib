"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from fairreckitlib.metrics.evaluator import Evaluator
from fairreckitlib.metrics.common import Metric


class EvaluatorLenskit(Evaluator):
    from lenskit import topn
    from lenskit.metrics import predict

    metricDict = {
        Metric.ndcg: topn.ndcg,
        Metric.precision: topn.precision,
        Metric.recall: topn.recall,
        Metric.mrr: topn.recip_rank,

        Metric.rmse: predict.rmse,
        Metric.mae: predict.mae,
    }

    # TODO refactor
    groupDict = {
        Metric.ndcg: 'ndcg',
        Metric.precision: 'precision',
        Metric.recall: 'recall',
        Metric.mrr: 'recip_rank',

        Metric.rmse: 'rmse',
        Metric.mae: 'mae'
    }

    topn_metrics = [Metric.ndcg, Metric.precision, Metric.recall, Metric.mrr]

    def load_test(self, test_path):
        self.test = pd.read_csv(test_path, header=None, sep='\t', names=['user', 'item', 'rating'])

    def load_train(self, train_path):
        self.train = pd.read_csv(train_path, header=None, sep='\t', names=['user', 'item', 'rating'])

    def load_recs(self, recs_path):
        recs = pd.read_csv(recs_path, header=None, sep='\t', names=['user', 'item', 'score'])
        recs['rank'] = recs.groupby('user')['score'].rank()
        recs['Algorithm'] = 'APPROACHNAME'
        self.recs = recs

    def evaluate(self):
        from lenskit import topn, metrics

        # evaluations = []
        # for metric in self.metrics:
        # TODO refactor self.metrics to metric?
        (metric, k) = self.metrics[0]
        eval_func = EvaluatorLenskit.metricDict[metric]
        print(eval_func)
        if metric in EvaluatorLenskit.topn_metrics:
            analysis = topn.RecListAnalysis()
            analysis.add_metric(eval_func, k=k)
            results = analysis.compute(self.recs, self.test).head()

            group_name = EvaluatorLenskit.groupDict[metric]
            evaluation = results.groupby('Algorithm')[group_name].mean()[0]
        else:
            from lenskit.metrics.predict import user_metric

            # TODO USER VS GLOBAL
            # TODO handle predictions without truth (missing)? Lenskit has 'ignore' or 'error'
            if metric in [Metric.rmse, Metric.mae]:
                # Merge on user ID
                scores = pd.merge(self.test, self.recs, how='left', on=['user','item'])
                scores.rename(columns={'score': 'prediction'}, inplace=True)
                print(scores)
                evaluation = user_metric(scores, metric=eval_func)
            else:
                raise Exception # Apparently there is another prediction metric that isn't handled

        return evaluation
        # evaluations.append(results[0])

        # return evaluations
