"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    Evaluates results from model algorithm
    """
    def __init__(self, eval_func, params, **kwargs):
        self.eval_func = eval_func
        self.params = params

    @abstractmethod
    def evaluate(self, train_set, test_set, recs):
        """Run analysis based on metric"""
        raise NotImplementedError()
