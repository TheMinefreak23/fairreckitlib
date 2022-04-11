"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from fairreckitlib.experiment import common
from .algorithms.factory import get_algorithm_list_from_factory, get_recommender_factory
from .data.registry import DataRegistry
from .experiment.run import run_experiment


class RecommenderSystem:
    """
    Top level API intended for use by applications
    """
    def __init__(self, data_dir, result_dir):
        self.data_registry = DataRegistry(data_dir)
        self.recommender_models = get_recommender_factory()

        self.result_dir = result_dir

        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

    def evaluate_experiment(self, experiment_dir, config):
        result_dir = os.path.join(self.result_dir, experiment_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        # evaluate additional metrics
        raise NotImplementedError()

    def run_experiment(self, config):
        result_dir = os.path.join(self.result_dir, config[common.EXP_KEY_NAME])
        if os.path.isdir(result_dir):
            raise IOError('Result already exists: ' + result_dir)

        os.mkdir(result_dir)

        run_0_dir = os.path.join(result_dir, 'run_0')
        os.mkdir(run_0_dir)

        run_experiment(
            run_0_dir,
            self.data_registry,
            config
        )

    def validate_experiment(self, experiment_dir, num_runs):
        result_dir = os.path.join(self.result_dir, experiment_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        # run the same experiment again for 'num_runs'
        raise NotImplementedError()

    def get_available_datasets(self):
        return self.data_registry.get_info()

    def get_available_predictors(self):
        raise NotImplementedError()

    def get_available_recommenders(self):
        recommenders = {}

        for rec_api in self.recommender_models:
            api_factory = self.recommender_models[rec_api]
            recommenders[rec_api] = get_algorithm_list_from_factory(api_factory)

        return recommenders

    def get_available_metrics(self):
        # TODO refactor
        from metrics.evaluator_lenskit import EvaluatorLenskit
        from metrics.evaluator_rexmex import EvaluatorRexmex
        EvaluatorLenskit.metricDict.keys() + EvaluatorRexmex.metricDict.keys()
        #raise NotImplementedError()
