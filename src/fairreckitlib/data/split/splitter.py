"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class DataSplitter(metaclass=ABCMeta):
    """Base class for FairRecKit data splitters.

    A splitter is used to split a dataframe into a train and test set.
    """
    def __init__(self, params):
        self.__params = params

    def get_params(self):
        """Get the parameters of the splitter.

        Returns:
            dict with the splitter parameters.
        """
        return dict(self.__params)

    @abstractmethod
    def run(self, dataframe, test_ratio):
        """Runs the splitter on the specified dataframe.

        Args:
            dataframe(pandas.DataFrame): with at least the 'user' column.
            test_ratio(float): the fraction of users to use for testing.

        Returns:
            train_set(pandas.DataFrame): the train set of the split.
            test_set(pandas.DataFrame): the test set of the split.
        """
        raise NotImplementedError()
