"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .random import RandomSplitter
from .temporal import TemporalSplitter

SPLIT_RANDOM = 'random'
SPLIT_TEMPORAL = 'temporal'

FUNC_CREATE_SPLIT = 'f_create_split'


class SplitFactory:
    """Split Factory with available data splitters."""
    def __init__(self):
        self.__factory = {}

    def add(self, split_name, func_create_split):
        """Adds a split method to the factory.

        This function raises a KeyError when the name of the method
        is already registered for this split factory.

        Args:
            split_name(str): name of the split method
            func_create_split(function): data splitter creation function.
        """
        if self.__factory.get(split_name) is not None:
            raise KeyError('Split method already exists: ' + split_name)

        self.__factory[split_name] = {FUNC_CREATE_SPLIT: func_create_split}

    def create(self, split_name):
        """Creates the split method with the specified name.

        This function raises a KeyError when the name of the split method
        is not registered for this split factory.

        Args:
            split_name(str): name of the split method.
        """
        if self.__factory.get(split_name) is None:
            raise KeyError('Split method does not exist: ' + split_name)

        func_create_split = self.__factory[split_name][FUNC_CREATE_SPLIT]
        split = func_create_split()
        split.name = split_name

        return split

    def get_available_split_names(self):
        """Gets the names of the available split methods in the factory.

        Returns:
            split_names(array like): list of split method names.
        """
        split_names = []

        for split_name, _ in self.__factory.items():
            split_names.append(split_name)

        return split_names

    def get_entries(self):
        """Gets the split method entries of the factory.

        Returns:
            factory entries(dict)
        """
        return dict(self.__factory)


def create_split_factory_from_list(split_list):
    """Creates a SplitFactory from a list of tuples.

    Each tuple consists of:

    1) split method name
    2) split creation function

    Args:
        split_list(array like): list of tuples.

    Returns:
        SplitFactory
    """
    factory = SplitFactory()

    for _, split in enumerate(split_list):
        (split_name, func_create_split) = split
        factory.add(
            split_name,
            func_create_split
        )

    return factory


def create_split_factory():
    """Creates a SplitFactory with all data splitters.

    Returns:
        (SplitFactory) with all splitters.
    """
    return create_split_factory_from_list([
        (SPLIT_RANDOM, _create_random_splitter),
        (SPLIT_TEMPORAL, _create_temporal_splitter)
    ])


def _create_random_splitter():
    """Creates the Random Splitter.

    Returns:
        (RandomSplitter) the random data splitter.
    """
    return RandomSplitter()


def _create_temporal_splitter():
    """Creates the Temporal Splitter.

    Returns:
        (TemporalSplitter) the temporal data splitter.
    """
    return TemporalSplitter()
