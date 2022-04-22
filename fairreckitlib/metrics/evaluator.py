"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import time
from abc import ABC, abstractmethod

from fairreckitlib.events import evaluation_event


class Evaluator(ABC):
    """
    Evaluates results from model algorithm
    """
    def __init__(self, train_path, test_path, recs_path, metrics, event_dispatcher):

        self.event_dispatcher = event_dispatcher

        #print('Loading train and test set..')
        self.test = self.load_test_process(test_path)
        self.train = self.load_train_process(train_path)
        #print(self.test.head())
        if not self.train.empty: print(self.train.head())  # TODO refactor
        #print('Train and test set loaded.')

        #print('Loading recs..')
        self.recs = self.load_recs_process(recs_path)  # TODO RENAME RECS
        #print(self.recs.head())
        #print('Recs loaded.')
        self.metrics = metrics

    def load_test_process(self, test_path):
        start=time.time()
        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_LOAD_TEST_SET,
            test_set_path=test_path
        )
        test = self.load_test(test_path)
        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_LOAD_TEST_SET,
            elapsed_time=time.time()-start
        )
        return test

    def load_train_process(self, train_path):
        start=time.time()
        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_LOAD_TRAIN_SET,
            train_set_path=train_path
        )
        train = self.load_train(train_path)
        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_LOAD_TRAIN_SET,
            elapsed_time=time.time()-start
        )
        return train

    def load_recs_process(self, recs_path):
        start=time.time()
        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_LOAD_RECS_SET,
            recs_set_path=recs_path
        )
        recs = self.load_recs(recs_path)
        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_LOAD_RECS_SET,
            elapsed_time=time.time()-start
        )
        return recs

    def evaluate_process(self):
        import time
        self.event_dispatcher.dispatch(
            evaluation_event.ON_BEGIN_EVAL,
            metric=self.metrics[0][0]
        )
        start = time.time()
        evaluation = self.evaluate()
        self.event_dispatcher.dispatch(
            evaluation_event.ON_END_EVAL,
            metric=self.metrics[0][0],
            elapsed_time=time.time()-start
        )
        return evaluation

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
