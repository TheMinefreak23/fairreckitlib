""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import elliot.run
import yaml

from ..evaluator import Evaluator

config_dir = 'config_files/'

class EvaluatorElliot(Evaluator):

    def __init__(self, metric, params):
        Evaluator.__init__(self, metric, params)

    def evaluate(self, test_set, recs):

        # Modify base proxy configuration.
        with open(config_dir + ) as file:
            data = yaml.safe_load(file)
            data['experiment']['evaluation'] = {'simple_metrics': metrics}
            data['experiment']['models'] = {'ProxyRecommender': {'path': rec_dir + path}}

        with open(config_dir + 'restore_rec.yml', 'w') as file:
            output = yaml.dump(data, file)

        # Evaluate the resulting recommendations.
        run_experiment(config_dir + 'restore_rec.yml')
        #TODO

class Metric:
    _name=''
    _category=''

class SimpleMetric(Metric):
    def configure(self):


class ComplexMetric(Metric):
    def configure(self):

def create_simple_metric(name):
    return EvaluatorElliot({},metric)

def create_simple_metric_k(params):
    return EvaluatorElliot(None, params)

def create_complex_metric(params):
    return EvaluatorElliot(None, params)