"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class RatingConverter(metaclass=ABCMeta):
    """Base class for converting ratings in a dataframe."""
    def __init__(self):
        pass

    @abstractmethod
    def run(self, df):
        raise NotImplementedError()
