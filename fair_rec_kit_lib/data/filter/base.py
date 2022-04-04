"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


"""
base class to filter a df (not a dataset in particular), as long as the df
contains a 'user' and 'item' column.

Dataset class is not really necessary here, but when combined with the
data.format module it needs to know about them to construct them.

Together with a factory pattern similar to the data.split module
we can define a variety of filters to exclude rows that do not satisfy the filter
from the specified df as long as it retains the 'user' and 'item' columns.

These filters could be used by the fair-rec-kit-app as well (table browsing).
"""
class DataFilter(metaclass=ABCMeta):

    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def run(self, df, **kwargs):
        raise NotImplementedError()
