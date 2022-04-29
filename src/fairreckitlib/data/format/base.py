"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


"""
base class to format a df (not a dataset in particular), as long as the df
contains a 'user' and 'item' column.

Dataset class will provide all relevant info on how the 'user' and/or 'item'
are connected to other files and thus new headers (columns).

Together with a factory pattern similar to the data.split module
we can define a variety of formatters to add/change/remove headers
present in the specified df as long as it retains the 'user' and 'item' columns.

These formatters could also be used by the data.filter module to avoid duplicate code.
Moreover they could be used by the fair-rec-kit-app as well (table browsing).
"""
class DataFormatter(metaclass=ABCMeta):

    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def run(self, df, **kwargs):
        raise NotImplementedError()
