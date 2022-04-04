"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base import DataSplitter


class TemporalSplitter(DataSplitter):

    def run(self, df, test_ratio, params):
        raise NotImplementedError()


def create_temporal_splitter():
    return TemporalSplitter()
