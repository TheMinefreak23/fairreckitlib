"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .random import RandomSplitter, create_random_splitter
from .temporal import TemporalSplitter, create_temporal_splitter

SPLIT_RANDOM = 'random'
SPLIT_TEMPORAL = 'temporal'


def get_split_factory():
    return {
        SPLIT_RANDOM: create_random_splitter,
        SPLIT_TEMPORAL: create_temporal_splitter
    }
    