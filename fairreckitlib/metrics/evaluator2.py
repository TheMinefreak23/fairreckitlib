from abc import ABC, abstractmethod
import pandas as pd

from fairreckitlib.metrics.metrics2 import Metric


class Evaluator(ABC):
    def __init__(self, train_path, test_path, recs_path, metrics):
        print('Loading train and test set..')
        self.load_test(test_path)
        self.load_train(train_path)
        print(self.test.head())
        if not self.train.empty: print(self.train.head())  # TODO refactor
        print('Train and test set loaded.')

        print('Loading recs..')
        self.load_recs(recs_path)  # TODO RENAME RECS
        print(self.recs.head())
        print('Recs loaded.')
        self.metrics = metrics

    @abstractmethod
    def load_test(self, test_path):
        """Loads test data from path"""
        raise NotImplementedError()

    @abstractmethod
    def load_train(self, train_path):
        """Loads train data from path"""
        raise NotImplementedError()

    @abstractmethod
    def load_recs(self, recs_path):
        """Loads recs data from path"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self):
        """Run analysis based on metric"""
        raise NotImplementedError()


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
        self.train = pd.DataFrame()  # Null, Lenskit doesn't need the train set

    def load_recs(self, recs_path):
        recs = pd.read_csv(recs_path, header=None, sep='\t', names=['user', 'item', 'score'])
        recs['rank'] = recs.groupby('user')['score'].rank()
        recs['Algorithm'] = 'APPROACHNAME'
        self.recs = recs

    def evaluate(self):
        from lenskit import topn, metrics

        #evaluations = []
        k = 10  # TODO get k from params
        #for metric in self.metrics:
        # TODO refactor self.metrics to metric?
        metric = self.metrics[0]
        eval_func = EvaluatorLenskit.metricDict[metric]
        print(eval_func)
        if metric in EvaluatorLenskit.topn_metrics:
            analysis = topn.RecListAnalysis()
            analysis.add_metric(eval_func, k=k)
            results = analysis.compute(self.recs, self.test).head()
        else:
            from lenskit.metrics.predict import user_metric

            # TODO USER VS GLOBAL
            if metric == Metric.rmse:
                results = user_metric(self.recs, metric=eval_func)
            if metric == Metric.mae:
                results = user_metric(self.recs, metric=eval_func)

        print(results)
        group_name = EvaluatorLenskit.groupDict[metric]
        results = results.groupby('Algorithm')[group_name].mean()
        print(results)
        print(results[0])
        return results[0]
            #evaluations.append(results[0])

        #return evaluations


class EvaluatorRexmex(Evaluator):
    from rexmex.metrics.ranking import normalized_discounted_cumulative_gain, average_precision_at_k, \
        average_recall_at_k, mean_reciprocal_rank
    from rexmex.metrics.rating import root_mean_squared_error, mean_absolute_error
    from rexmex.metrics.coverage import item_coverage
    from rexmex.metrics import intra_list_similarity, novelty

    metricDict = {
        Metric.ndcg: normalized_discounted_cumulative_gain,
        Metric.precision: average_precision_at_k,
        Metric.recall: average_recall_at_k,
        Metric.mrr: mean_reciprocal_rank,
        Metric.rmse: root_mean_squared_error,
        Metric.mae: mean_absolute_error,
        Metric.item_coverage: item_coverage,
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
        # Merge on user ID
        scores = pd.merge(self.recs, self.test[['source_id', 'y_true']], on=['source_id'])
        print(scores.head())

        #evaluations = []
        k = 10  # TODO get k from params
        #for metric in self.metrics:
        # TODO refactor self.metrics to metric?
        metric = self.metrics[0]
        eval_func = EvaluatorRexmex.metricDict[metric]
        print(eval_func)
        # TODO refactor
        if metric == Metric.ndcg:
            evaluation = eval_func()
        elif metric == Metric.precision:
            evaluation = eval_func(scores['source_id'], scores['y_true'], k)
        elif metric == Metric.mae:
            evaluation = eval_func(scores['y_true'], scores['y_score'])
        elif metric == Metric.item_coverage:
            possible_users_items, tuple_recs = self.prepare_for_coverage()
            evaluation = eval_func(possible_users_items, tuple_recs)
        elif metric == Metric.user_coverage:
            evaluation = eval_func(possible_users_items, tuple_recs)
        print(evaluation)
        return evaluation
        #evaluations.append(evaluation)

        #return evaluations

    def prepare_for_coverage(self):
        # user, item columns to list of tuples
        tuple_recs = [tuple(r) for r in self.recs[['source_id', 'target_id']].to_numpy()]
        print(tuple_recs[0:10])
        possible_users_items = (self.train['source_id'], self.train['target_id'])
        return possible_users_items, tuple_recs
