""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABC, abstractmethod


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


