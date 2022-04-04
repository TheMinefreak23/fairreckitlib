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
        #TODO

class Metric:
    _name=''
    _category=''

class SimpleMetric(Metric):
    def configure(self):
            #TODO

class ComplexMetric(Metric):
    def configure(self):
            #TODO

def create_metric(params):
    return EvaluatorElliot(None,params)

def create_metric_ndcg(params):
    return EvaluatorElliot(None,params)

def create_metric_precision(params):
    return EvaluatorElliot(None,params)

def create_metric_recall(params):
    return EvaluatorElliot(None,params)

def create_metric_map(params):
    return EvaluatorElliot(None,params)

def create_metric_mar(params):
    return EvaluatorElliot(None,params)

def create_simple_metric(params):
    return EvaluatorElliot(None,params)

def create_simple_metric_k(params):
    return EvaluatorElliot(None, params)

def create_complex_metric(params):
    return EvaluatorElliot(None, params)